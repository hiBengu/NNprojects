import pandas as pd
import numpy as np
import pickle

# Read excel
dataFrame = pd.read_excel(
    "Siparisler_31072020144940.xls",      # relative python path to subdirectory
    sep=',',         # Tab-separated value file.
    skiprows=0,         # Skip the first 10 rows of the file
)
print("Excel Dosyasi okumasi tamamlandi!")

# Sadece ekmekleri sec: 15070 ==> 6088
dataFrame = dataFrame[dataFrame['Kategori'] == 'Çeşnili Ekmekler']

dataFrame = dataFrame[['Teslimat Günü','Barkod']]
print("Gerekli Sutunlar Elde Edildi!")

new = dataFrame["Teslimat Günü"].str.split(".", n = 2, expand = True)

dataFrame['Gun'] = new[0]
dataFrame['Ay'] = new[1]
dataFrame['Yil'] = new[2]
dataFrame.drop(columns =["Teslimat Günü"], inplace = True)

dataFrame['Teslimat Tarihi'] = dataFrame['Yil'].str.cat(dataFrame['Ay'],sep="-")
dataFrame['Teslimat Tarihi'] = dataFrame['Teslimat Tarihi'].str.cat(dataFrame['Gun'],sep="-")

dataFrame['Teslimat Tarihi'] = pd.to_datetime(dataFrame['Teslimat Tarihi'], errors='coerce')
dataFrame['Haftanin Gunu'] = dataFrame['Teslimat Tarihi'].dt.dayofweek
dataFrame.drop(columns =["Teslimat Tarihi"], inplace = True)
print("Tarih Ayristirildi!")

dataFrame['Barkod'], barkodNo = pd.factorize(dataFrame['Barkod'])
print("Urunlere Numaralar verildi!")
print(barkodNo)

# String columnlari float'a cevirildi
dataFrame['Gun'] = dataFrame['Gun'].astype(float)
dataFrame['Ay'] = dataFrame['Ay'].astype(float)
dataFrame['Yil'] = dataFrame['Yil'].astype(float)
dataFrame['Haftanin Gunu'] = dataFrame['Haftanin Gunu'].astype(float)
dataFrame['Barkod'] = dataFrame['Barkod'].astype(float)

# Sadece 2020 yili verileri alindi 6088 ==> 5854
dataFrame = dataFrame[dataFrame['Yil'] == 2020]
dataFrame = dataFrame[dataFrame['Barkod'] > 0]
dataFrame.drop(columns=['Yil'], inplace=True)

print(dataFrame.groupby(['Gun', 'Ay'])['Barkod'].apply(list).reset_index())
dataFrame = dataFrame.groupby(['Gun', 'Ay','Haftanin Gunu'])['Barkod'].apply(list).reset_index()

trainData = dataFrame[dataFrame['Ay'] < 7].copy()
testData = dataFrame[dataFrame['Ay'] == 7].copy()

# Normalizasyon Islemleri
trainData['Gun'] /= max(trainData['Gun'])
trainData['Ay'] /= max(trainData['Ay'])
trainData['Haftanin Gunu'] /= max(trainData['Haftanin Gunu'])

testData['Gun'] /= max(testData['Gun'])
testData['Ay'] /= max(testData['Ay'])
testData['Haftanin Gunu'] /= max(testData['Haftanin Gunu'])

trainY = trainData['Barkod']
trainData.drop(columns=['Barkod'], inplace=True)

testY = testData['Barkod']
testData.drop(columns=['Barkod'], inplace=True)

print("Training DataFrame:")
print(trainData)
print(trainY)

print("Test DataFrame")
print(testData)
print(testY)

# Dataset numpy'a cevirildi
trainData = trainData.to_numpy()
trainY = trainY.to_numpy()
testData = testData.to_numpy()
testY = testY.to_numpy()

trainData = trainData.astype(np.float32)
trainY = np.array(trainY)
testData = testData.astype(np.float32)
testY = np.array(testY)

for i, row in enumerate(trainY):
    row.append(10.0)
    trainY[i] = np.bincount(row)
    trainY[i].astype(np.float32)
    trainY[i] = trainY[i][:-1]

for i, row in enumerate(testY):
    row.append(10.0)
    testY[i] = np.bincount(row)
    testY[i].astype(np.float32)
    testY[i] = testY[i][:-1]

print(trainData)
print(trainY)

print("Last Part")
print(testY.shape)
trainY = np.stack(trainY, axis=0)
testY = np.stack(testY, axis=0)
print(testY.shape)

# Save data as pickle
with open('xDataTrain.pkl', 'wb') as f:
            pickle.dump(trainData, f)

with open('yDataTrain.pkl', 'wb') as f:
            pickle.dump(trainY, f)

with open('xDataTest.pkl', 'wb') as f:
            pickle.dump(testData, f)

with open('yDataTest.pkl', 'wb') as f:
            pickle.dump(testY, f)
