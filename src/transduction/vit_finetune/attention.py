import torch
import numpy as np
import cv2

from torchvision.transforms import ToPILImage
transform_to_pil = ToPILImage()

from ..plot_if import get_plt
plt = get_plt()


# FYI
#---- ^^ https://github.com/huggingface/pytorch-image-models/discussions/1232
def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward
#---- $$


#---- ^^ https://gist.github.com/zlapp/40126608b01a5732412da38277db9ff5

def get_mask(im, att_mat):
    # Average the attention weights across all heads.
    # att_mat,_ = torch.max(att_mat, dim=1)
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    if 0:  #==== @@ orig
        mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
        result = (mask * im).astype("uint8")

        #print(result.shape, joint_attentions.shape, grid_size)
        # (224, 224, 3) torch.Size([12, 197, 197]) 14

        return result, joint_attentions, grid_size
    if 1:  #==== @@ !!
        mask = cv2.resize(mask / mask.max(), im.size)
        print('@@ mask.shape:', mask.shape)  # (224, 224)
        return mask, joint_attentions, grid_size
        #====
        # result = (mask[..., np.newaxis] * im).astype("uint8")
        # print('@@ result.shape:', result.shape)  # (224, 224, 3)
        #return result, joint_attentions, grid_size

#---- $$


def verify_attentions(model, testds, ckpt_file=None):

    for idx, x in enumerate(testds):
        print(idx, x['img'], x['label'], x['pixels'].shape)

        input = x['pixels']
        input_path = x['img']
        print('@@ input.shape:', input.shape)  # torch.Size([3, 224, 224])

        outputs = model(input.unsqueeze(0), output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions
        print('@@ logits:', logits)
        print('@@ type(attentions):', type(attentions))  # <class 'tuple'>
        for i, attn in enumerate(attentions):
            print(f'@@ attn[{i}]: {attn.shape}')

        #

        im_input = transform_to_pil(input)
        mask, joint_attentions, grid_size = get_mask(im_input, torch.cat(attentions))

        print(f'@@ testds[{idx}]: path={input_path}')
        im_orig_resized = cv2.resize(plt.imread(input_path), mask.shape)
        #print('@@ im_orig_resized.shape:', im_orig_resized.shape)  # (224, 224, 3)

        # TODO refactor into a method
        #---- ^^ mask --> heat attention
        mask_stacked = torch.tensor([mask[:,:]], dtype=torch.float32)
        #print('@@ mask_stacked.shape:', mask_stacked.shape)  # torch.Size([1, 224, 224])

        mask_stacked = torch.stack([mask_stacked], dim=0)
        #print('@@ mask_stacked.shape:', mask_stacked.shape)  # -> torch.Size([1, 1, 224, 224])

        from ...wsdan.net.augment import generate_heatmap
        heat_attention_map = generate_heatmap(mask_stacked)

        raw_image = torch.tensor([im_orig_resized[:,:]], dtype=torch.float32).permute(0, 3, 1, 2)
        #print('@@ raw_image.shape:', raw_image.shape)  # torch.Size([1, 3, 224, 224])

        heat_attention_image = (raw_image * 0.3) + (heat_attention_map.cpu() * 0.7)
        #print('@@ heat_attention_image.shape:', heat_attention_image.shape)  # torch.Size([1, 3, 224, 224])

        heat_attention_image = transform_to_pil(heat_attention_image[0])
        #heat_attention_image.save(f'heat_attention_{idx}.png')
        #---- $$

        #----
        fig = plt.figure()

        axes = []
        rows, cols = 1, 3

        axes.append(fig.add_subplot(rows, cols, 1))
        plt.imshow(im_orig_resized, cmap='gray')

        axes.append(fig.add_subplot(rows, cols, 2))
        plt.imshow(mask, cmap='gray')

        axes.append(fig.add_subplot(rows, cols, 3))
        plt.imshow(heat_attention_image)

        fig.suptitle(f'testds[{idx}] | attention_mask_{idx} | heat_attention_{idx}\n'
                     f'(path: {input_path})\n'
                     f'(ViT model: {ckpt_file})')

        plt.axis('off')
        plt.setp(axes, xticks=[], yticks=[])
        # assume `mkdir inference`
        plt.savefig(f'inference/attention_mask_{idx}_{ckpt_file}.png', bbox_inches='tight')
        #----

