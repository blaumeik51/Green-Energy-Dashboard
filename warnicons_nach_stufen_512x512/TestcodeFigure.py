import matplotlib.patches as patches
# gererate axis label for for weather data
timeVecWeather=[]
rain=[]
for hour in range(0,len(localWeather)):
    a=dt.datetime.fromtimestamp(localWeather['dt'][hour])
    timeVecWeather.append(a.strftime('%d') + '.' + a.strftime('%m') + '. ' + a.strftime('%H') + ':' + a.strftime('%M'))
    try:
       rain.append(localWeather['rain'][hour]['3h'])
    except:
       rain.append(0)
       
timeVecWeather=[]
for hour in range(0,len(localWeather)):
    a=dt.datetime.fromtimestamp(localWeather['dt'][hour])
    timeVecWeather.append(a.strftime('%H'))
    #timeVecWeather.append(a.strftime('%d') + '.' + a.strftime('%m') + '. ' + a.strftime('%H') + ':' + a.strftime('%M'))

# calculate colors for energy
energyColor=[]
for i in range(0,len(localWeather)):
    energyColor.append(ledColorFun(localWeather['energyTemp'][i],'energy'))


fig = plt.figure(figsize=(10, 10))
mng = plt.get_current_fig_manager()

plt.style.use('dark_background')
plt.rcParams['axes.linewidth'] = 0.1
plt.suptitle(now.strftime('%H') + ':' + now.strftime('%M') + ' Uhr',size=50)

# Weather data text
ax=fig.add_subplot(4,2,1)
plt.subplot2grid((4, 2), (0, 0), rowspan=2)
plt.plot(range(0,10),range(0,10),color='black')
plt.text(0, 7,'Temperatur: ', ha='left', va='center',color='white',fontsize=25)
plt.text(5, 7, str(int(localWeather['temp'][0])) + ' °C', ha='left', va='center',color='white',fontsize=25)

plt.text(0, 5,'gefühlt wie:', ha='left', va='center',color='white',fontsize=20)
plt.text(5, 5, str(int(localWeather['tempFeelsLike'][0])) + ' °C', ha='left', va='center',color='white',fontsize=20)

plt.text(0, 4,'Luftfeuchtigkeit:', ha='left', va='center',color='white',fontsize=20)
plt.text(5, 4,str(int(localWeather['humidity'][0])) + ' %', ha='left', va='center',color='white',fontsize=20)

try:
    plt.text(0, 3,'Regen:', ha='left', va='center',color='white',fontsize=20)
    plt.text(5, 3,str(localWeather['rain'][0]['3h']) + ' mm', ha='left', va='center',color='white',fontsize=20)
except:
    plt.text(0, 3,'Regen:', ha='left', va='center',color='white',fontsize=20)
    plt.text(5, 3,'0 mm', ha='left', va='center',color='white',fontsize=20)

plt.text(0, 0,'Sonnenaufgang:', ha='left', va='center',color='white',fontsize=20)
plt.text(5, 0,localSunrise.strftime('%H') + ':' + localSunrise.strftime('%M') + ' Uhr', ha='left', va='center',color='white',fontsize=20)

plt.text(0, 1,'Sonnenuntergang:', ha='left', va='center',color='white',fontsize=20)
plt.text(5, 1,localSunset.strftime('%H') + ':' + localSunset.strftime('%M') + ' Uhr', ha='left', va='center',color='white',fontsize=20)


plt.axis('off')

# Temperaur plot
ax=fig.add_subplot(4,2,5)

plt.plot(range(0,len(localWeather['temp'])*3,3),localWeather['temp'],'royalblue',
          range(0,len(localWeather['tempFeelsLike'])*3,3),localWeather['tempFeelsLike'], 'grey')
plt.fill_between(range(0,len(localWeather['tempFeelsLike'])*3,3),localWeather['tempFeelsLike'], alpha=0.4,color='grey')
if max(localWeather['tempFeelsLike'])<30 and min(localWeather['tempFeelsLike'])>0:
    plt.ylim(0, 30)
plt.xlim(0,len(localWeather['tempFeelsLike'])*3)
plt.title('Temperatur',ha='center')
plt.legend(['Temperatur','gefühlte Temperatur'],loc='upper right')
plt.grid(alpha=0.2)
ax.set_ylabel('Temperatur [°C]',color='grey')
ax.set_xlabel('Stunden')
plt.grid(alpha=0.2)
ax2=ax.twinx()
plt.bar(range(0,len(localWeather['temp'])*3,3),rain, width=3, bottom=None, align='edge', color='royalblue')
ax2.set_ylabel('Regen [mm]',color='royalblue')

if max(rain)<10:
    plt.ylim(0, 10)
plt.xlim(0,len(localWeather['tempFeelsLike'])*3)
 
# cloud and wind speed
ax=fig.add_subplot(4,2,7)
plt.plot(range(0,len(localWeather['cloud_cov'])*3,3),localWeather['cloud_cov'], 'grey')
plt.fill_between(range(0,len(localWeather['cloud_cov'])*3,3),localWeather['cloud_cov'], alpha=0.4,color='grey')
ax.set_ylabel('Bewölkung [%]',color='grey')
ax.set_xlabel('Stunden')
plt.xlim(0,len(localWeather['wind_speed'])*3)
plt.grid(alpha=0.2)
ax2=ax.twinx()
plt.plot(range(0,len(localWeather['wind_speed'])*3,3),localWeather['wind_speed'],'royalblue')
ax2.set_ylabel('Windgeschwindigkeit [m/s]',color='royalblue')


plt.xlim(0,len(localWeather['cloud_cov'])*3)

# Control
fig.add_subplot(424)
plt.plot(range(0,len(data)),np.divide(range(0,len(data)),len(data))*4,color='black',alpha=0)
ax=fig.add_subplot(424)

# ventilation
Idx=3
if vent_red:
    plt.scatter(0,Idx,400,color='green',edgecolor='none')
else:
    plt.scatter(0,Idx,400,color='red',edgecolor='none')

# compressor
Idx=2
for i in range(0,compressorHours):
    ax.add_patch(patches.Rectangle((dataControl['index'][i], -.25 + Idx), 1, .5,color='g'))
if compressor:
    plt.scatter(0,Idx,400,color='green',edgecolor='none')
else:
    plt.scatter(0,Idx,400,color='red',edgecolor='none')
    
# addHeating
Idx=1
for i in range(0,addHeatingHours):
    ax.add_patch(patches.Rectangle((dataControl['index'][i], -.25 + Idx), 1, .5,color='g'))
if addHeating:
    plt.scatter(0,Idx,400,color='green',edgecolor='none')
else:
    plt.scatter(0,Idx,400,color='red',edgecolor='none')
  
# Auto  
Idx=0
for i in range(period[0],period[0] + chargingHours):
    ax.add_patch(patches.Rectangle((i, -.25 + Idx), 1, .5,color='g'))
if car:
    plt.scatter(0,Idx,400,color='green',edgecolor='none')
else:
    plt.scatter(0,Idx,400,color='red',edgecolor='none')


plt.ylim(-0.5, 3.5)
plt.grid(alpha=0.2)
ax.set_yticks(range(0,4))
ax.set_yticklabels(['Auto laden optimal','Zusatzheizung aktiv','Wärmepumpe aktiv','Lüftung reduziert'])

# Market data
fig.add_subplot(426)
#plt.scatter(range(0,len(data)),data['marketprice'],c=ledColor[16:16+len(data)],edgecolor='none',linewidths=40)
#plt.fill_between(range(0,len(data)),data['marketprice'],color=ledColor[16:16+len(data)], alpha=0.2)
plt.bar(range(0,len(data)),data['marketprice'], width=0.8, bottom=None, align='edge', color=ledColor[16:16+len(data)])
plt.title('Börsenpreis',ha='center')
plt.xlabel('Stunden')
plt.ylabel('€/MWh')
plt.grid(alpha=0.2)

# Energy plot
fig.add_subplot(428)
#subplot_kw=dict(frameon=False)
#plt.scatter(range(0,24,3),localWeather['energyTemp'][0:8],c=ledColor[49:57],edgecolor='none',linewidths=40)
plt.bar(range(0,120,3),localWeather['energyTemp'],color=energyColor,width=3 ,bottom=None, align='edge')
#plt.bar(range(0,120,3),localWeather['energyTemp'],color=ledColor[49:57],bottom=None, align='center')
plt.title('Erneuerbare Energien Index',ha='center')
plt.grid(alpha=0.2)
plt.xlabel('Stunden')
plt.ylabel('[-]')
#plt.xticks(timeVecWeather)
#plt.fill_between(range(0,24,3),localWeather['energyTemp'][0:8],color=ledColor[49:57], alpha=0.2)
fig.tight_layout()

plt.pause(.1)

# X-Achse auf Stunden bringen...