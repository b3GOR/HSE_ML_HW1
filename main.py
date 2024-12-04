# Загрузка обученной модели
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import re
import numpy as np
import io
from fastapi.responses import StreamingResponse
from joblib import load
import pandas as pd
import numpy as np
import re

app=FastAPI()

model_for_csv = load('L2(best_weight).pkl')
pipeline=load('pipeline.pkl')


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]
def preprocess(item:Items):

    item['mileage'] = item['mileage'].replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)
    item['engine'] = item['engine'].replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)
    item['max_power'] = item['max_power'].replace(r'[^\d.]', '', regex=True).replace('', np.nan).astype(float)

    def extract_torque(x):
        x = str(x).strip()
        if '@' in x:
            y = x.split('@')[0].strip()  
        elif 'at' in x:
            y = x.split('at')[0].strip()  
        else:
            y = x.strip()  
        if 'kgm' in x or '(kgm' in x:

            numbers = re.findall(r'\d+\.\d+', y)
            if numbers:
                return round(float(numbers[0]) * 9.81, 2)  
            else:
                numbers = re.findall(r'\d+', y)
                if numbers:
                    return round(float(numbers[0]) * 9.81, 2)  

        if '-' in y:
            range_values = [float(i.replace(',', '')) for i in y.split('-')]
            return round(sum(range_values) / len(range_values), 2)

        numbers = re.findall(r'\d+\.\d+', y)
        if numbers:
            return float(numbers[0])  
        else:
            numbers = re.findall(r'\d+', y)
            if numbers:
                return float(numbers[0])  

        return None           
    def extract_max_torque_rpm(x):
        x = str(x).strip()
        x = x.replace(',', '')

        rpm_value = None
        if '@' in x or 'at' in x:
            if '@' in x:
                rpm_part = x.split('@')[-1]
            else:
                rpm_part = x.split('at')[-1]
            rpm_range = re.findall(r'(\d{1,4})\D*(\d{1,4})rpm', rpm_part)
            if rpm_range:
                rpm_min, rpm_max = map(int, rpm_range[0])
                rpm_value = (rpm_min + rpm_max) // 2  
            else:
                rpm_single = re.findall(r'(\d+)\s*rpm', rpm_part)
                if rpm_single:
                    rpm_value = int(rpm_single[0]) 

        return rpm_value

    item['max_torque_rpm'] = item['torque'].copy()
    item['torque'] = item['torque'].apply(extract_torque)
    item['max_torque_rpm'] = item['max_torque_rpm'].apply(extract_max_torque_rpm)
    item.fillna(item.select_dtypes(include='number').median(), inplace=True)

    item['engine'] = item['engine'].astype('int64')
    item['seats'] = item['seats'].astype('int64')

    item['name'] = item['name'].apply(lambda x: 'Land Rover' if x == 'Land' else x.split()[0])
    means = pd.read_csv('MeanTarget.csv', index_col=0)


    item['name'] = item['name'].map(means['selling_price']).fillna(means['selling_price'].mean())
    item=pd.get_dummies(item,columns=[ 'fuel', 'seller_type', 'transmission', 'owner'],drop_first=True,dtype=int)

    return item




@app.post('/csv')
async def csv_pred(data: UploadFile = File(description="Загрузите CSV-файл для обработки")):
    df = pd.read_csv(data.file)
    df=preprocess(df)
    result = round(model_for_csv.predict(df),0)
    df['selling_price'] = result
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                 )
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response


@app.post('/json')
async def json_pred(data: UploadFile=File(description="Загрузите json для обработки")):
    df = pd.read_json(data.file)
    df=preprocess(df)
    df=df.drop('name',axis=1)
    result = pipeline.predict(df)
    return round(float(-result),0)



