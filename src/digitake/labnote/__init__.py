import pytz
import gspread
from datetime import datetime
from oauth2client.client import GoogleCredentials

def open_google_sheet(filename, sheet_no):
  gc = gspread.authorize(GoogleCredentials.get_application_default())
  sheets = gc.open(filename).worksheets()
  return sheets[sheet_no]

def post_result_to_sheet(worksheet, model_name, results):
  tz_bkk = pytz.timezone('Asia/Bangkok')
  data = [datetime.now(tz=tz_bkk).strftime("%Y-%m-%d %H:%M"), model_name] + results
  r = worksheet.append_row(data)
  return r

#x = post_result_to_sheet(worksheet, "DenseNet161", [0.1234, 12.34, 99.99])
#print(x)
