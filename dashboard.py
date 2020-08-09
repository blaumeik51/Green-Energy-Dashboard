import datetime as dt
import requests
import time
import numpy as np
import numexpr
import pandas as pd
from simple_dwd_weatherforecast import dwdforecast
from suntime import Sun
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib import cm
import locale
import platform
import os
from solarpy import solar_panel
import warnings
warnings.filterwarnings("ignore")

if platform.system() != 'Windows':
    import RPi.GPIO as GPIO

pd.set_option("display.max_rows", None, "display.max_columns", None)
locale.setlocale(locale.LC_ALL, '')

# initialize location
# Ahrensburg
lat = 53.67
lon = 10.24

# initialize figures
plt.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure(" ", figsize=(19.88910891089109, 10.28095238095238))
fig.set_tight_layout(False)

# initialize solar panel
panel = solar_panel(1, 0.2, id_name='NYC_xmas')  # surface, efficiency and name
panel.set_orientation(np.array([0, 0, -1]))  # upwards
panel.set_position(lat, lon, 0)  # NYC latitude, longitude, altitude
panel.set_datetime(dt.datetime(2019, 12, 25, 13, 15))  # Christmas Day!
panel.power()

y=0.415
x=0.03564904697110492
dx=0.4181903129732449/244
startHour=-9
size=0.05
ax421a = plt.subplot2grid((4, 2), (0, 0), rowspan=2)  # Weather data text
ax421b = fig.add_axes([0.25, 0.8, 0.2, 0.2], anchor='NE', zorder=-1) # weather symbol
ax421c = fig.add_axes([dx*(startHour+24), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421d = fig.add_axes([dx*(startHour+48), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421e = fig.add_axes([dx*(startHour+72), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421f = fig.add_axes([dx*(startHour+96), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421g = fig.add_axes([dx*(startHour+120), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421h = fig.add_axes([dx*(startHour+144), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421i = fig.add_axes([dx*(startHour+168), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421j = fig.add_axes([dx*(startHour+192), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421k = fig.add_axes([dx*(startHour+216), y, size, size], anchor='NE', zorder=1) # weather symbol
ax421l = fig.add_axes([dx*(startHour+240), y, size, size], anchor='NE', zorder=1) # weather symbol
ax422 = fig.add_subplot(422)  # Control
ax424 = fig.add_subplot(424)  # Control
ax425 = fig.add_subplot(425)
ax425b = ax425.twinx()
ax426 = fig.add_subplot(426)
ax427 = fig.add_subplot(427)  # cloud and wind speed
ax427b = ax427.twinx()
ax428 = fig.add_subplot(428)  # Energy forcast plot

# figManager.full_screen_toggle()

# GPIO definition
addHeatingGPIO = 26  # Relay 1 cable yellow
compressorGPIO = 19  # Relay 2 cable green
ventilationGPIO = 13  # Relay 3 cable blue
waterTempIncreaseGPIO = 6  # Relay 4 cable purple

if platform.system() != 'Windows':
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(waterTempIncreaseGPIO, GPIO.OUT)
    GPIO.setup(ventilationGPIO, GPIO.OUT)
    GPIO.setup(compressorGPIO, GPIO.OUT)
    GPIO.setup(addHeatingGPIO, GPIO.OUT)

    GPIO.output(waterTempIncreaseGPIO, GPIO.LOW)
    GPIO.output(ventilationGPIO, GPIO.LOW)
    GPIO.output(compressorGPIO, GPIO.LOW)
    GPIO.output(addHeatingGPIO, GPIO.LOW)


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


# set definitions

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def isActivationAllowed(now, numHours, timePeriod, data, contiguous):
    if numHours > 0:
        if contiguous:  # if identifiend period shall be contiguous as for car charging
            period = []
            eeValue = data['eevalue']
            for idx in range(0, timePeriod - numHours + 1):
                period.append(sum(eeValue[idx:idx + numHours]))
            period = pd.DataFrame(period).idxmax()
            start = data.iloc[period[0]]['startTime']
            end = data.iloc[period[0] + numHours - 1]['endTime']
            if now >= start and now < end:
                return True
            else:
                return False
        else:  # non contiguous period identification
            data = data.loc[:timePeriod - 1]
            data = data.sort_values('eevalue', ascending=False).reset_index(drop=True)
            start_times = data.loc[:numHours - 1, 'startTime']
            end_times = data.loc[:numHours - 1, 'endTime']
            for idx in range(0, numHours):
                if now >= start_times[idx] and now < end_times[idx]:
                    return True
            return False
        return False
    return False


def setDataControl(now, data, dataControl, numHours, timePeriod, columnName, contiguous, initial):
    for hour in range(0, len(data)):
        div = np.divide(data['startTime'][0].hour + hour, timePeriod)
        if div.is_integer() or initial:
            if hour + timePeriod <= len(data):
                for nextHour in range(0, timePeriod):
                    dataControl.loc[hour + nextHour, columnName] = isActivationAllowed(
                        dataControl.loc[hour + nextHour, 'startTime'], numHours, timePeriod,
                        data.loc[hour:hour + timePeriod - 1].reset_index(), contiguous)

    return dataControl


def setDataControl2(now, data, dataControl, columnNameIn, columnNameOut, threshold, operator):
    # set values according to thresholds
    for hour in range(0, min([len(data), len(dataControl)])):
        if operator == '>=':
            if data.loc[hour, columnNameIn] >= threshold:
                if dataControl.loc[hour, 'compressor'] == False:
                    dataControl.loc[hour, columnNameOut] = True
                else:
                    dataControl.loc[hour, columnNameOut] = False
        elif operator == '<=':
            if data.loc[hour, columnNameIn] <= threshold:
                if dataControl.loc[hour, 'compressor'] == False:
                    dataControl.loc[hour, columnNameOut] = True
                else:
                    dataControl.loc[hour, columnNameOut] = False
    return dataControl


def energyFromWind(wind_speed):
    if wind_speed <= 12 and wind_speed > 3:  # wind turbiens operating
        return wind_speed ** 3 / (12 ** 3) * 100
    elif wind_speed <= 3 or wind_speed > 25:  # wind turbiens idling or too much wind
        return 0
    elif wind_speed > 12:  # wind turbiens fullload
        return 100


def energyFromSolar(cloud_cov, time):
    panel.set_datetime(time)
    cloud_importance = 0.5  # im 100% cloudy, the energy from PV is reduced by 50%
    return (panel.power()/158*(1-cloud_cov/100 * cloud_importance))*100


def energyColorFun(value, indicator):
    if indicator == 'price':
        if value >= 40:  # red
            return [1, 0, 0]
        elif value >= 10 and value < 40:  # orange
            return [1, 1 - value / 40, 0]
        elif value >= 0 and value < 10:  # green
            return [value / 10, 1, 0]
        elif value < 0:  # blue
            return [0, 0, 1]

    elif indicator == 'energy':
        if value <= 30:  # redish
            return [1, value / 100, 0]
        elif value > 30 and value <= 70:  # orange
            return [1, value / 100 * 1.428, 0]
        elif value > 70 and value < 120:  # yellow
            return [1 - value / 100 + 0.3, 1, 0]
        elif value >= 120:  # green
            return [0, 1, 0]

    elif indicator == 'temperature':
        if value >= 25:  # redish
            return [1, 0, 0]
        elif value < 25 and value >= 15:  # orange
            return [1, value / 100 * 1.428, 0]
        elif value > 15 and value < 30:  # orange
            return [1 - value / 100 + 0.3, 1, 0]
        elif value >= 30:  # red
            return [1, 0, 0]


def getDataFromURL(url, time):
    for tries in range(0,3):
        try:
            w = requests.get(url)
            if w.status_code == 200:
                print
                return w, True
            else:
                time.sleep(60)
        except:
            time.sleep(60)
    print(time.asctime() + ': Failed to load data from url: ' + url)
    return [], False

def addWeatherPreview(ax,os,condition):
    try:
        ax.cla()
        ax.axis('off')
        if max(condition.to_list())>5+most_frequent(condition.to_list()):
            cond = max(condition.to_list())
        else:
            cond = most_frequent(condition.to_list())
        path = os.path.join(os.getcwd(), 'wettericons', str(int(cond)) + '.png')
        #path = os.path.join(os.getcwd(), 'wettericons', '29.png')
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.axis('off')
        return ax
    except:
        print('could not load: ' + path)
        
# define variables
now = dt.datetime.now()
utcnow = dt.datetime.utcnow()
utcoffset = now.hour - utcnow.hour

sun = Sun(lat, lon)  # initalize
sunrise = sun.get_sunrise_time() + dt.timedelta(hours=utcoffset)
sunset = sun.get_sunset_time() + dt.timedelta(hours=utcoffset)

dataControl = pd.DataFrame()
weatherData = pd.DataFrame()
weatherDataDaily = pd.DataFrame()
localWeather = []
dataPrevious = []
initial = True
lastWaterTempIncrease = now + dt.timedelta(days=7)
loop = True
while loop:
    # for j in range(0,1):
    now = dt.datetime.now()
    if (now.minute == 2) or initial:  # every hour the display shall be update
        # get energy data from corrently API
        result = getDataFromURL('https://api.corrently.io/core/gsi?zip=22926', time)
        if result[1]:
            energyData = result[0].json()['forecast']
            energyData = pd.DataFrame.from_records(energyData)
            for hour in range(0, len(energyData)):
                energyData.loc[hour, 'startTime'] = dt.datetime.fromtimestamp(energyData.loc[hour, 'epochtime'])
                energyData.loc[hour, 'endTime'] = dt.datetime.fromtimestamp(
                    energyData.loc[hour, 'epochtime']) + dt.timedelta(hours=1)
            # Append data set, with actual hour as the API sataset starts with t+1h
            if isinstance(dataPrevious, list):
                energyData = energyData.append(energyData.loc[0], ignore_index=True)
                energyData.loc[energyData.index[-1], 'startTime'] = energyData.loc[0, 'startTime'] - dt.timedelta(
                    hours=1)
                energyData.loc[energyData.index[-1], 'endTime'] = energyData.loc[0, 'endTime'] - dt.timedelta(hours=1)
            else:
                energyData = energyData.append(
                    dataPrevious.loc[energyData['startTime'] == dt.datetime(now.year, now.month, now.day, now.hour)],
                    ignore_index=True)
            energyData = energyData.sort_values(by='startTime').reset_index(drop=True)
            
        # get weather data and calculate energy from wind and solar from dwd
        div = np.divide(now.hour, 6)  # new forecasts are issued every 6 hours
        if div.is_integer() or initial:
            dwd_weather = dwdforecast.Weather("10147")  # Flughafen Hamburg
            dwd_weather.update()
            weatherData = pd.DataFrame(dwd_weather.forecast_data).T.reset_index()
            weatherData = weatherData.rename(columns={'index': 'original_time'})

            for hour in range(0, len(weatherData)):
                weatherData.loc[hour, 'start_time'] = dt.datetime.strptime(weatherData.loc[hour, 'original_time'][:13],
                                                                           '%Y-%m-%dT%H') + dt.timedelta(
                    hours=utcoffset)
                weatherData.loc[hour, 'end_time'] = dt.datetime.strptime(weatherData.loc[hour, 'original_time'][:13],
                                                                         '%Y-%m-%dT%H') + dt.timedelta(
                    hours=utcoffset + 1)
                weatherData.loc[hour, 'sun_cov'] = weatherData.loc[hour, 'sun_dur'] / 60 * 100
                weatherData.loc[hour, 'wind_speed'] = float(weatherData.loc[hour, 'wind_speed'])
                weatherData.loc[hour, 'wind_speed_Kmh'] = weatherData.loc[hour, 'wind_speed'] * 3.6
                weatherData.loc[hour, 'cloud_cov'] = float(weatherData.loc[hour, 'cloud_cov'])
                weatherData.loc[hour, 'cloud_covNum'] = weatherData.loc[hour, 'cloud_cov'] * 2 / 2
                weatherData.loc[hour, 'sun_prop'] = 100 - weatherData.loc[hour, 'cloud_covNum']
                weatherData.loc[hour, 'day'] = weatherData.loc[hour, 'start_time'].day
                weatherData.loc[hour, 'tempNum'] = weatherData.loc[hour, 'temp'] / 2 * 2
                weatherData.loc[hour, 'prec_sum'] = float(weatherData.loc[hour, 'prec_sum'])
                try:
                    weatherData.loc[hour, 'condition_num'] = float(weatherData.loc[hour, 'condition'])
                except:
                    weatherData.loc[hour, 'condition_num'] = 99

            # get weather Data from open weather 
            result = getDataFromURL(
                'https://api.openweathermap.org/data/2.5/forecast?lat=53.67&lon=10.24&appid=8c66e2d516fac691ea64523271da3f73',
                time)
            if result[1]:
                weatherDataOpDa = pd.DataFrame.from_records(result[0].json()['list'])
                weatherDataOpDa['start_time'] = weatherDataOpDa.dt.apply(dt.datetime.fromtimestamp)
                weatherDataOpDa['cloud_cov'] = weatherDataOpDa.clouds.apply(lambda x: x['all'])
                weatherDataOpDa['wind_speed'] = weatherDataOpDa.wind.apply(lambda x: x['speed'])
                weatherDataOpDa['tempK'] = weatherDataOpDa.main.apply(lambda x: x['temp'])
                weatherDataOpDa['mintemp'] = weatherDataOpDa.main.apply(lambda x: x['temp_min']) - 273.15
                weatherDataOpDa['maxtemp'] = weatherDataOpDa.main.apply(lambda x: x['temp_max']) - 273.15
                weatherDataOpDa['pressure'] = weatherDataOpDa.main.apply(lambda x: x['pressure'])
                weatherDataOpDa['humidity'] = weatherDataOpDa.main.apply(lambda x: x['humidity'])
                weatherDataOpDa['tempFeelsLike'] = weatherDataOpDa.main.apply(lambda x: x['feels_like']) - 273.15
                weatherDataOpDa['temp'] = weatherDataOpDa['tempK'] - 273.15

        # clean for past values
        weatherData = weatherData.drop(
            weatherData[weatherData.start_time < now - dt.timedelta(hours=1)].index).reset_index(drop=True)
        weatherDataOpDa = weatherDataOpDa.drop(
            weatherDataOpDa[weatherDataOpDa.start_time < now - dt.timedelta(hours=1)].index).reset_index(drop=True)
        energyData = energyData.drop(energyData[energyData.startTime < now - dt.timedelta(hours=1)].index).reset_index(
            drop=True)

        # wind
        energyBasedonWeather = weatherData.loc[:, 'wind_speed'].apply(energyFromWind)

        # biomass
        energyBasedonWeather = energyBasedonWeather + 30

        # solar
        for idx in weatherData.index:
            energyBasedonWeather[idx] += energyFromSolar(weatherData.loc[idx, 'cloud_cov'], weatherData.loc[idx, 'start_time']) *0.9

        # scaling values to fit with data from API
        energyBasedonWeather = energyBasedonWeather * 0.8

        # get market prices from awattar
        result = getDataFromURL('https://api.awattar.de/v1/marketdata?', time)
        if result[1]:
            stockData = result[0].json()['data']
            stockData = pd.DataFrame.from_records(stockData)
            for hour in range(0, len(stockData)):
                stockData.loc[hour, 'start_time'] = dt.datetime.fromtimestamp(
                    float(str(stockData.loc[hour, 'start_timestamp'])[0:10]))
                stockData.loc[hour, 'end_time'] = dt.datetime.fromtimestamp(
                    float(str(stockData.loc[hour, 'end_timestamp'])[0:10]))

        # clean for past values
        stockData = stockData.drop(stockData[stockData.start_time < now - dt.timedelta(hours=1)].index).reset_index(
            drop=True)

        # determine number of heating operation hours for the next 24h
        meanTemp = weatherData.loc[:24, 'temp'].mean()

        if meanTemp > 17:
            addHeatingHours = [0, 24]
            compressorHours = [2, 24]
        elif meanTemp > 10 and meanTemp <= 17:
            addHeatingHours = [0, 24]
            compressorHours = [5, 24]
        elif meanTemp <= 10 and meanTemp >= 5:
            addHeatingHours = [3, 24]
            compressorHours = [24, 24]
        elif meanTemp < 5 and meanTemp >= -5:
            addHeatingHours = [4, 24]
            compressorHours = [24, 24]
        elif meanTemp < -5:
            addHeatingHours = [6, 24]
            compressorHours = [24, 24]

        thresVent = 22  # lower ventilation, if outside temperature is larger than value
        thresPrice = 0  # allow all consumers to run, if price is lower than valuee
        
        if initial: #initialization of the dataControl matrix
            dataControl = pd.DataFrame()
        else: # delete times, which are older than now
            dataControl = dataControl.drop(dataControl[dataControl.startTime < now - dt.timedelta(hours=1)].index).reset_index(
            drop=True)

        # extend the df for the next 48 hours
        for hour in range(len(dataControl), 48):
            if len(dataControl)!=0:
                nowTemp = dataControl['startTime'].iloc[-1]
                timedelta=1
            else:
                nowTemp=now
                timedelta=0
                
            dataControl.loc[hour, 'startTime'] = dt.datetime(nowTemp.year, nowTemp.month, nowTemp.day,
                                                             nowTemp.hour) + dt.timedelta(hours=timedelta)
            dataControl.loc[hour, 'endTime'] = dt.datetime(nowTemp.year, nowTemp.month, nowTemp.day,
                                                           nowTemp.hour) + dt.timedelta(hours=timedelta+1)

            dataControl.loc[hour, 'compressor'] = False
            dataControl.loc[hour, 'addHeating'] = False
            dataControl.loc[hour, 'car'] = False
            dataControl.loc[hour, 'ventilation'] = False
            dataControl.loc[hour, 'waterTempIncrease'] = False
        
        if energyData.index.stop>len(dataControl): #ensure that energyData is not longer than dataControl
            energyData = energyData.drop(energyData[energyData.index.values>47].index)
                
        if initial == False:  # not calculated on initial start of the computation
            dataControl = setDataControl2(now, stockData, dataControl, 'marketprice', 'compressor', thresPrice,
                                          '<=')  # either the market price is negative or the eevalue is high
            dataControl = setDataControl2(now, stockData, dataControl, 'marketprice', 'addHeating', thresPrice,
                                          '<=')  # either the market price is negative or the eevalue is high
            dataControl = setDataControl2(now, stockData, dataControl, 'marketprice', 'car', thresPrice,
                                          '<=')  # either the market price is negative or the eevalue is high

        dataControl = setDataControl(now, energyData, dataControl, compressorHours[0], min([len(energyData), compressorHours[1]]), 'compressor',
                                     False, initial)
        dataControl = setDataControl(now, energyData, dataControl, addHeatingHours[0], min([len(energyData), addHeatingHours[1]]), 'addHeating',
                                     False, initial)
        dataControl = setDataControl(now, energyData, dataControl, 4, min([len(energyData), len(dataControl)]), 'car',
                                     True, initial)
        dataControl = setDataControl2(now, weatherData, dataControl, 'tempNum', 'ventilation', thresVent, '>=')
        dataControl = setDataControl2(now, stockData, dataControl, 'marketprice', 'waterTempIncrease', thresPrice, '<=')

        # ensure regualar water temperatrue increase
        if lastWaterTempIncrease < now:
            dataControl = setDataControl(now, energyData, dataControl, 1, len(energyData), 'waterTempIncrease',
                                         True)  # ensure regualar water temperatrue increase

        # ensure that if water temperature increase is true that also addHeating, ventialition and compressor is set accordingly
        for hour in range(0, len(dataControl)):
            nowTemp = dataControl.loc[hour, 'startTime']
            if nowTemp.hour == 22 or nowTemp.hour == 23:  # reduce ventilation when going to bed
                dataControl.loc[hour, 'ventilation'] = True
            if dataControl.loc[hour, 'waterTempIncrease']:
                dataControl.loc[hour, 'ventilation'] = False
                dataControl.loc[hour, 'compressor'] = True
                dataControl.loc[hour, 'addHeating'] = True

                # ensure that heating is allowed in initial programm start
        if initial:
            for hour in range(0, 2):
                dataControl.loc[hour, 'compressor'] = True
                dataControl.loc[hour, 'addHeating'] = True
                dataControl.loc[hour, 'ventilation'] = False

                # plot dashboard
        # gererate axis label for for weather data    
        timeVecWeather = []
        timeVecWeatherLabel = []
        timeVecWeatherMinor = []
        timeVecWeatherLabel = []
        for hour in range(0, len(weatherData)):
            a = weatherData.loc[hour, 'start_time']
            if a.hour == 0:
                timeVecWeather.append(hour)
                timeVecWeatherLabel.append(
                    ' ' + a.strftime('%a') + ' ' + a.strftime('%d') + '.' + a.strftime('%m') + '.')
            if (weatherData.loc[hour, 'start_time'].hour / 3).is_integer():
                timeVecWeatherMinor.append(hour)

        # gererate axis label for for control data    
        timeVecControl = []
        timeVecControlLabel = []
        for hour in range(0, len(dataControl)):
            if (dataControl.loc[hour, 'startTime'].hour / 3).is_integer():
                a = dataControl.loc[hour, 'startTime']
                timeVecControl.append(hour)
                if a.hour == 0:
                    timeVecControlLabel.append(a.strftime('%d') + '.' + a.strftime('%m'))
                else:
                    timeVecControlLabel.append(a.strftime('%H'))

        # gererate axis label for for energy data    
        timeVecEnergy = []
        timeVecEnergyLabel = []
        for hour in range(0, len(energyData)):
            if (energyData.loc[hour, 'startTime'].hour / 3).is_integer():
                a = energyData.loc[hour, 'startTime']
                timeVecEnergy.append(hour)
                if a.hour == 0:
                    timeVecEnergyLabel.append(a.strftime('%d') + '.' + a.strftime('%m'))
                else:
                    timeVecEnergyLabel.append(a.strftime('%H'))

        # calculate colors for energy
        energyColor = []
        for i in range(0, len(energyBasedonWeather)):
            energyColor.append(energyColorFun(energyBasedonWeather[i], 'energy'))

        # calculate colors for energy
        stockColor = []
        for i in range(0, len(stockData)):
            stockColor.append(energyColorFun(stockData.loc[i, 'marketprice'], 'price'))

        eeValueColor = []
        for i in range(0, len(energyData)):
            eeValueColor.append(energyColorFun(energyData.loc[i, 'eevalue'], 'energy'))

        # highlight periods with negative prices with blue color
        for i in range(0, min([len(stockData), len(eeValueColor)])):
            if stockData.loc[i, 'marketprice'] < 0:
                eeValueColor[i] = [0, 0, 1]

        ## plotting dashboard
        # plt.figure(2)
        ax421a.cla()
        column1 = -1400
        column2 = -500
        ax421a.set_xlim(column1, 470)
        ax421a.set_ylim(-400, 512)
        row1 = 480
        row2 = 250
        row3 = 150
        row4 = 50
        row5 = -50
        row6 = -280
        row7 = -380
        # ax421a.plot(range(0, 10), range(0, 10), color='black', alpha=1)
        try:
            ax421b.cla()
            ax421b.axis('off')
            # weather picture showing the most frequent occuring weather within the next 6 hours
            if now.timestamp()>sunrise.timestamp() and now.timestamp()<sunset.timestamp(): #daytime
                path = os.path.join(os.getcwd(), 'wettericons',
                                str(int(most_frequent(weatherData.loc[:5, 'condition_num'].to_list()))) + '.png')
            else: #night time
                path = os.path.join(os.getcwd(), 'wettericons',
                                str(int(most_frequent(weatherData.loc[:5, 'condition_num'].to_list()))) + 'n.png')
                if os.path.exists(path) == False:
                    path = os.path.join(os.getcwd(), 'wettericons',
                                str(int(most_frequent(weatherData.loc[:5, 'condition_num'].to_list()))) + '.png')
      
            #path = os.path.join(os.getcwd(), 'wettericons','30.png')
            img = mpimg.imread(path)
            ax421b.imshow(img)
        except:
            print('Could not load: ' + path)
        try:
            addWeatherPreview(ax421c,os,weatherData.loc[6:30, 'condition_num'])
            addWeatherPreview(ax421d,os,weatherData.loc[30:54, 'condition_num'])
            addWeatherPreview(ax421e,os,weatherData.loc[54:78, 'condition_num'])
            addWeatherPreview(ax421f,os,weatherData.loc[78:92, 'condition_num'])
            addWeatherPreview(ax421g,os,weatherData.loc[78:92, 'condition_num'])
            addWeatherPreview(ax421h,os,weatherData.loc[92:116, 'condition_num'])
            addWeatherPreview(ax421i,os,weatherData.loc[116:140, 'condition_num'])
            addWeatherPreview(ax421j,os,weatherData.loc[140:164, 'condition_num'])
            addWeatherPreview(ax421k,os,weatherData.loc[188:202, 'condition_num'])
            addWeatherPreview(ax421l,os,weatherData.loc[202:236, 'condition_num'])
        except:
            ''
                
        ax421a.grid(alpha=0.2)
        ax421a.text(column1, row1, 'Temperatur:', ha='left', va='center', color='white', fontsize=35)
        ax421a.text(column2, row1, str(weatherData.loc[0, 'temp']) + ' °C', ha='left', va='center', color='white',
                    fontsize=35)

        ax421a.text(column1, row2, 'gefühlt wie:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row2, str(int(weatherDataOpDa.loc[0, 'tempFeelsLike'] * 10) / 10) + ' °C', ha='left',
                    va='center',
                    color='white', fontsize=20)

        ax421a.text(column1, row3, 'Luftfeuchtigkeit:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row3, str(weatherDataOpDa.loc[0, 'humidity']) + ' %', ha='left', va='center',
                    color='white',
                    fontsize=20)

        ax421a.text(column1, row4, 'Regen:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row4,
                    str(int(weatherData.loc[0, 'prec_sum'] * 100) / 100) + ' l/' + '$m^{2}$' + ' (' + str(
                        int(sum(weatherData.loc[:24, 'prec_sum']) * 100) / 100) + ' l/' + '$m^{2}$' + ' in 24h)',
                    ha='left',
                    va='center', color='white', fontsize=20)

        ax421a.text(column1, row5, 'Windgeschwindigkeit:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row5, str(weatherData.loc[0, 'wind_speed']) + ' m/s' + ' (' + str(
            int(weatherData.loc[0, 'wind_speed'] * 3.6)) + ' km/h)', ha='left', va='center', color='white', fontsize=20)

        ax421a.text(column1, row6, 'Sonnenaufgang:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row6, sunrise.strftime('%H') + ':' + sunrise.strftime('%M') + ' Uhr', ha='left',
                    va='center',
                    color='white', fontsize=20)

        ax421a.text(column1, row7, 'Sonnenuntergang:', ha='left', va='center', color='white', fontsize=20)
        ax421a.text(column2, row7, sunset.strftime('%H') + ':' + sunset.strftime('%M') + ' Uhr', ha='left', va='center',
                    color='white', fontsize=20)

        ax421a.axis('off')

        # Temperaure plot
        ax425.cla()
        ax425b.cla()
        cmap = mpl.cm.Spectral
        norm = mpl.colors.Normalize(vmin=-10, vmax=30)
        COL = MplColorHelper('RdYlBu_r', -10, 35)
        color = []
        for i in range(0, len(weatherData['tempNum'])):
            color.append(COL.get_rgb(weatherData.loc[i, 'tempNum']))
        # ax425.scatter(range(0, len(weatherData['tempNum'])), weatherData['tempNum'], c=weatherData['tempNum'], cmap='Spectral_r', norm=norm)
        ax425.bar(range(0, len(weatherData['tempNum'])), weatherData['tempNum'], color=color, alpha=0.6, width=1,
                  bottom=None)
        ax425.plot(range(0, len(weatherData['tempNum'])), weatherData['tempNum'],color='grey',linewidth=2)

        # 425.fill_between(range(0, len(weatherData['tempNum'])), weatherData['tempNum'], alpha=0.5, cmap=cmap)
#         if max(weatherData['tempNum']) < 30 and min(weatherData['tempNum']) > 0:
#             ax425.set_ylim(0, 30)
            
        ax425.set_ylim(ax425.get_ylim()[0],abs(ax425.get_ylim()[1])+abs(abs(ax425.get_ylim()[1]))*0.4)
        ax425.set_yticks(ax425.get_yticks()[(np.where(ax425.get_yticks()<max(weatherData['tempNum']) )[0].tolist())].tolist())
        ax425.grid(alpha=0.2)
        ax425.set_ylabel('Temperatur [°C]')
        ax425.set_xticks(timeVecWeather, minor=False)
        ax425.set_xticklabels(timeVecWeatherLabel, rotation=0, minor=False, horizontalalignment='left')
        ax425b.set_xticks(timeVecWeatherMinor, minor=True)
        ax425b.bar(range(0, len(weatherData['prec_sum'])), weatherData['prec_sum'], width=3, bottom=None, align='edge',
                   color='green')
        ax425b.set_ylabel('Regen [mm]', color='green')
        ax425b.set_xlim(0, len(weatherData['tempNum']))
        ax425b.set_title('Vorhersage Temperatur', ha='center')
        ax425b.set_xticks(timeVecWeather, minor=False)
        ax425b.set_xticklabels(timeVecWeatherLabel, rotation=0, minor=False, horizontalalignment='left')
        ax425b.set_xticks(timeVecWeatherMinor, minor=True)
        if max(weatherData['prec_sum']) < 5:
            lim = 5
            ax425b.set_ylim(0, lim)
        else:
            lim = max(weatherData['prec_sum'])
        ax425b.set_ylim(ax425b.get_ylim()[0],abs(ax425b.get_ylim()[1])+abs(abs(ax425b.get_ylim()[1]))*0.4)
        ax425b.set_yticks(ax425b.get_yticks()[(np.where(ax425b.get_yticks()<lim )[0].tolist())].tolist())

        # wind and clouds
        ax427.cla()
        ax427.plot(range(0, len(weatherData['cloud_cov'])), weatherData['sun_prop'], 'yellow', alpha=0.3)
        ax427.fill_between(range(0, len(weatherData['sun_prop'])), weatherData['sun_prop'], alpha=0.3,
                           color='yellow')
        #ax425.bar(range(0, len(weatherData['tempNum'])), weatherData['tempNum'], color=color, alpha=0.6, width=1,bottom=None)
        ax427.set_title('Vorhersage Sonneschein und Wind', ha='center')
        ax427.set_ylabel('Sonneschein [%]', color='yellow', alpha=0.7)
        ax427.set_xlim(0, len(weatherData['wind_speed']))
        ax427.set_ylim(0, 100)
        ax427.set_xticks(timeVecWeather, minor=False)
        ax427.set_xticklabels(timeVecWeatherLabel, rotation=0, horizontalalignment='left')
        ax427.set_xticks(timeVecWeatherMinor, minor=True)
        ax427.grid(alpha=0.2)
        ax427b.cla()
        ax427b.plot(range(0, len(weatherData['wind_speed_Kmh'])), weatherData['wind_speed_Kmh'], 'royalblue')
        ax427b.fill_between(range(0, len(weatherData['wind_speed_Kmh'])), weatherData['wind_speed_Kmh'], alpha=0.4,
                            color='royalblue')
        ax427b.set_ylabel('Windgeschwindigkeit [km/h]', color='royalblue')
        ax427b.set_xticks(timeVecWeather, minor=False)
        ax427b.set_xticklabels(timeVecWeatherLabel, rotation=0, horizontalalignment='left')
        ax427b.set_xticks(timeVecWeatherMinor, minor=True)

        if max(weatherData['wind_speed']) < 100:
            ax427b.set_ylim(0, 100)
        ax427.set_xlim(0, len(weatherData))

        # EE information
        ax422.cla()
        ax422.plot(range(0, 10), range(0, 10), color='black')

        ax422.text(0, 8, 'CO' '$_{2}$' + ' Emmissionen: ', ha='left', va='center', color='white', fontsize=20)
        ax422.text(6, 8, str(energyData.loc[0, 'co2_g_standard']) + ' g/kWh', ha='left', va='center', color='white',
                   fontsize=20)

        ax422.text(0, 3.5, 'Anteil erneuerbare Energien: ', ha='left', va='center', color='white', fontsize=20)
        ax422.text(6, 3.5, str(energyData.loc[0, 'eevalue']) + ' %', ha='left', va='center', color='white', fontsize=20)

        ax422.axis('off')

        # Control
        ax424.cla()
        ax424.plot(range(0, len(dataControl)), np.divide(range(0, len(dataControl)), len(dataControl)) * 4,
                   color='black', alpha=0)
        ax424.set_title('Gerätesteuerung', ha='center')
        ax424.set_xlim(-1, len(dataControl))
        ax424.set_xticks(timeVecControl, minor=False)
        ax424.set_xticklabels(timeVecControlLabel, rotation=0, horizontalalignment='center')
        ax424.set_xticks(range(0, len(dataControl)), minor=True)
        ax424.set_xlabel('Uhrzeit')
        ax424.set_ylim(-0.5, 4.5)
        ax424.grid(alpha=0.2)
        ax424.set_yticks(range(0, 5))
        ax424.set_yticklabels(
            ['Auto laden', 'Zusatzheizung aktiv', 'Wärmepumpe aktiv', 'Lüftung reduziert', 'Brauchwasser erhöhen'])

        # Auto  
        Idx = 0
        for i in range(0, len(dataControl)):
            if dataControl.loc[i, 'car']:
                ax424.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5, color='g'))
        if dataControl.loc[0, 'car']:
            ax424.scatter(0, Idx, 400, color='green', edgecolor='none')
        else:
            ax424.scatter(0, Idx, 400, color='red', edgecolor='none')

        # addHeating
        Idx = 1
        for i in range(0, len(dataControl)):
            if dataControl.loc[i, 'addHeating']:
                ax424.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5, color='g'))
        if dataControl.loc[0, 'addHeating']:
            ax424.scatter(0, Idx, 400, color='green', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(addHeatingGPIO, GPIO.HIGH)
        else:
            ax424.scatter(0, Idx, 400, color='red', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(addHeatingGPIO, GPIO.LOW)

        # compressor
        Idx = 2
        for i in range(0, len(dataControl)):
            if dataControl.loc[i, 'compressor']:
                ax424.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5, color='g'))
        if dataControl.loc[0, 'compressor']:
            ax424.scatter(0, Idx, 400, color='green', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(compressorGPIO, GPIO.HIGH)
        else:
            ax424.scatter(0, Idx, 400, color='red', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(compressorGPIO, GPIO.LOW)

        # ventilation
        Idx = 3
        for i in range(0, len(dataControl)):
            if dataControl.loc[i, 'ventilation']:
                ax424.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5, color='g'))
        if dataControl.loc[0, 'ventilation']:
            ax424.scatter(0, Idx, 400, color='green', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(ventilationGPIO, GPIO.HIGH)
        else:
            ax424.scatter(0, Idx, 400, color='red', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(ventilationGPIO, GPIO.LOW)

        # water temperature increase
        Idx = 4
        for i in range(0, len(dataControl)):
            if dataControl.loc[i, 'waterTempIncrease']:
                ax424.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5, color='g'))
        if dataControl.loc[0, 'waterTempIncrease']:
            ax425.scatter(0, Idx, 400, color='green', edgecolor='none')
            lastWaterTempIncrease = now + dt.timedelta(days=7)
            if platform.system() != 'Windows':
                GPIO.output(waterTempIncreaseGPIO, GPIO.HIGH)
        else:
            ax424.scatter(0, Idx, 400, color='red', edgecolor='none')
            if platform.system() != 'Windows':
                GPIO.output(waterTempIncreaseGPIO, GPIO.LOW)

        # EE value
        ax426.cla()
        ax426.bar(range(0, len(energyData)), energyData['eevalue'], width=0.8, bottom=None, align='edge',
                  color=eeValueColor, alpha=0.6)
        ax426.set_title('Anteil erneuerbare Energien', ha='center')
        ax426.grid(alpha=0.2)
        ax426.set_ylabel('Anteil [%]')
        ax426.set_xlabel('Uhrzeit')
        ax426.set_xticks(timeVecEnergy, minor=False)
        ax426.set_xticklabels(timeVecEnergyLabel, rotation=0, horizontalalignment='center')
        ax426.set_xticks(range(0, len(energyData)), minor=True)
        ax426.set_xlim(-1, len(dataControl))
        ax426.set_ylim(0, 100)
        # ax426.set_xlim(0,len(energyData))

        # Energy forcast plot
        ax428.cla()
        ax428.bar(range(0, len(energyBasedonWeather)), energyBasedonWeather, color=energyColor, width=1, bottom=None, alpha=0.5)
        ax428.plot(range(0, len(energyBasedonWeather)), energyBasedonWeather,color='grey',linewidth=2)
        ax428.set_title('Vorhersage Anteil erneuerbare Energien', ha='center')
        ax428.grid(alpha=0.2)
        ax428.set_ylabel('Anteil [%]')
        ax428.set_xlim(0, len(energyBasedonWeather))
        ax428.set_ylim(0, 100)
        ax428.set_xticks(timeVecWeather, minor=False)
        ax428.set_xticklabels(timeVecWeatherLabel, rotation=0, horizontalalignment='left')
        initial = False
        fig.tight_layout()

        plt.pause(.1)
        if platform.system() != 'Windows':
            time.sleep(60)
        else:
            plt.savefig('dashboard.png',bbox_inches='tight')

        dataControl.to_json('dataControl.json')
        energyData.to_json('energyData.json')
        weatherData.to_json('weatherData.json')
        stockData.to_json('stockData.json')
        weatherDataOpDa.to_json('weatherDataOpDa.json')
    
    if platform.system() != 'Windows':
        time.sleep(30)
    else:
        loop = False
