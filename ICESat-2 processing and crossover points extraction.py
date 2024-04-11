# -*- coding: gb18030 -*-
import csv
import datetime
import glob
import math
import os
import re
import shutil
import time

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import tqdm
from numba import jit
from osgeo import ogr
from osgeo import osr
from tqdm import tqdm


def timechange(TIME):
    time = float(TIME)
    days = time // 86400
    start_date = '2018-01-01'
    d = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    delta = datetime.timedelta(days=days)
    n_days = d + delta
    newdays = n_days.strftime('%Y-%m-%d')
    return newdays

def clip_shapefiles(input_shapefile, clip_shapefile, output_folder):

    clip = gpd.read_file(clip_shapefile)
    for file in os.listdir(input_shapefile):
        if file.endswith(".shp"):

            input_file = os.path.join(input_shapefile, file)
            gdf = gpd.read_file(input_file)

            clipped = gpd.overlay(gdf, clip, how='intersection')

            output_file = os.path.join(output_folder, f"{file}")

            clipped.to_file(output_file)


# Folder path of h5 file
source_folder = r'E:\新程序\新建文件夹 (2)'

'''
Data classification
'''

if not os.path.exists(source_folder + '\\' + '+x'):
    os.makedirs(source_folder + '\\' + '+x')

if not os.path.exists(source_folder + '\\' + '-x'):
    os.makedirs(source_folder + '\\' + '-x')

# 新文件+x文件夹路径
positive_folder = source_folder + '\\' + '+x'
# 新文件-x文件夹路径
negative_folder = source_folder + '\\' + '-x'

# 多个日期区间
date_ranges = [
    ('20180916', '20181228'),
    ('20190907', '20200514'),
    ('20210115', '20211002'),
    ('20220609', '20230209'),
    # You can continue to add more date ranges
]


for filename in os.listdir(source_folder):

    match = re.search(r'\d{8}', filename)
    if match:
        file_date = match.group()
        if any(start_date <= file_date <= end_date for start_date, end_date in date_ranges):
            shutil.move(os.path.join(source_folder, filename), os.path.join(positive_folder, filename))
        else:
            shutil.move(os.path.join(source_folder, filename), os.path.join(negative_folder, filename))

print('Classification of documents completed')

'''
data extraction
'''

PATH_right = positive_folder   # H5文件路径
filename = os.listdir(PATH_right)

guidao = {'gt1r', 'gt2r', 'gt3r'}

for i in guidao:
    lat_lujing = i + "/land_ice_segments/latitude"
    lon_lujing = i + "/land_ice_segments/longitude"
    h_lujing = i + "/land_ice_segments/h_li"
    time_lujing = i + "/land_ice_segments/delta_time"
    quality_lujing = i + "/land_ice_segments/atl06_quality_summary"

    for s in filename:
        da = h5py.File(PATH_right + "\\" + s)

        latitude = da.get(lat_lujing)
        longitude = da.get(lon_lujing)
        h_li = da.get(h_lujing)
        delta_time = da.get(time_lujing)
        quality = da.get(quality_lujing)

        lat = np.array(latitude)
        lon = np.array(longitude)
        h = np.array(h_li)
        time = np.array(delta_time)
        quality_summary06 = np.array(quality)

        # ATL06
        ds = pd.DataFrame(
            {'lon': lon, 'lat': lat, 'ele': h, 'delta_time': time, 'quality_summary06': quality_summary06})
        ele = ds['ele']


        if not os.path.exists(PATH_right + "\\" + i + "\\" + "csv"):
            os.makedirs(PATH_right + "\\" + i + "\\" + "csv")

        ds.to_csv(PATH_right + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", index=False, header=True)

        # 修改时间
        df = pd.read_csv(PATH_right + "\\" + i + "\\" + "csv" + "\\" + s + ".csv",
                         converters={'delta_time': timechange})
        df.to_csv(PATH_right + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", encoding="utf_8_sig", index=False, mode='w')

        df1 = pd.read_csv(PATH_right + "\\" + i + "\\" + "csv" + "\\" + s + ".csv")  # index_col = 0
        row_indexs = df1[df1["quality_summary06"] == 1].index
        df1.drop(row_indexs, inplace=True)
        df1.to_csv(PATH_right + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", encoding="utf_8_sig", index=False,
                   mode='w')

        if not os.path.exists(PATH_right + "\\" + i + "\\" + "shp"):
            os.makedirs(PATH_right + "\\" + i + "\\" + "shp")

        ogr.UseExceptions()


        shp_path = PATH_right + "\\" + i + "\\" + "shp"

        csv_path = PATH_right + "\\" + i + "\\" + "csv"

    for csv_filename in glob.glob(os.path.join(csv_path, '*.csv')):

        # 读入csv文件信息，设置点几何的字段属性
        csv_df = pd.read_csv(csv_filename)

        driver = ogr.GetDriverByName('ESRI Shapefile')

        shp_filename = os.path.basename(csv_filename)[:-7] + '.shp'

        if os.path.exists(os.path.join(shp_path, shp_filename)):
            driver.DeleteDataSource(os.path.join(shp_path, shp_filename))
        ds = driver.CreateDataSource(os.path.join(shp_path, shp_filename))

        layer_name = os.path.basename(csv_filename)[:-7]

        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)

        out_lyr = ds.CreateLayer(layer_name, srs=sr, geom_type=ogr.wkbPoint)


        # ICESst ATL06
        lon_fld = ogr.FieldDefn('lon', ogr.OFTReal)
        lon_fld.SetWidth(50)
        lon_fld.SetPrecision(15)
        out_lyr.CreateField(lon_fld)
        # Latitude
        lat_fld = ogr.FieldDefn('lat', ogr.OFTReal)
        lat_fld.SetWidth(50)
        lat_fld.SetPrecision(15)
        out_lyr.CreateField(lat_fld)
        # ele
        ele_fld = ogr.FieldDefn('ele', ogr.OFTReal)
        ele_fld.SetWidth(50)
        ele_fld.SetPrecision(10)
        out_lyr.CreateField(ele_fld)
        # delta_time
        time_fld = ogr.FieldDefn('time', ogr.OFTString)
        time_fld.SetWidth(50)
        out_lyr.CreateField(time_fld)
        # quality_summary06
        quality_fld = ogr.FieldDefn('quality', ogr.OFTInteger)
        quality_fld.SetWidth(50)
        out_lyr.CreateField(quality_fld)


        featureDefn = out_lyr.GetLayerDefn()
        feature = ogr.Feature(featureDefn)


        point = ogr.Geometry(ogr.wkbPoint)


        for i in range(len(csv_df)):
            # ICESat-2
            feature.SetField('lon', float(csv_df.iloc[i, 0]))
            feature.SetField('lat', float(csv_df.iloc[i, 1]))
            feature.SetField('ele', float(csv_df.iloc[i, 2]))
            feature.SetField('time', str(csv_df.iloc[i, 3]))
            feature.SetField('quality', float(csv_df.iloc[i, 4]))

            # 利用经纬度创建点， X为经度lon， Y为纬度lat
            point.AddPoint(float(csv_df.iloc[i, 0]), float(csv_df.iloc[i, 1]))
            feature.SetGeometry(point)


            out_lyr.CreateFeature(feature)

        ds.Destroy()

PATH_left = negative_folder
filename = os.listdir(PATH_left)

guidao = {'gt1l', 'gt2l', 'gt3l'}

for i in guidao:
    lat_lujing = i + "/land_ice_segments/latitude"
    lon_lujing = i + "/land_ice_segments/longitude"
    h_lujing = i + "/land_ice_segments/h_li"
    time_lujing = i + "/land_ice_segments/delta_time"
    quality_lujing = i + "/land_ice_segments/atl06_quality_summary"

    for s in filename:
        da = h5py.File(PATH_left + "\\" + s)

        latitude = da.get(lat_lujing)
        longitude = da.get(lon_lujing)
        h_li = da.get(h_lujing)
        delta_time = da.get(time_lujing)
        quality = da.get(quality_lujing)

        lat = np.array(latitude)
        lon = np.array(longitude)
        h = np.array(h_li)
        time = np.array(delta_time)
        quality_summary06 = np.array(quality)

        # ATL06
        ds = pd.DataFrame(
            {'lon': lon, 'lat': lat, 'ele': h, 'delta_time': time, 'quality_summary06': quality_summary06})
        ele = ds['ele']


        if not os.path.exists(PATH_left + "\\" + i + "\\" + "csv"):
            os.makedirs(PATH_left + "\\" + i + "\\" + "csv")

        ds.to_csv(PATH_left + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", index=False, header=True)

        df = pd.read_csv(PATH_left + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", converters={'delta_time': timechange})
        df.to_csv(PATH_left + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", encoding="utf_8_sig", index=False, mode='w')

        df1 = pd.read_csv(PATH_left + "\\" + i + "\\" + "csv" + "\\" + s + ".csv")  # index_col = 0
        row_indexs = df1[df1["quality_summary06"] == 1].index
        df1.drop(row_indexs, inplace=True)
        df1.to_csv(PATH_left + "\\" + i + "\\" + "csv" + "\\" + s + ".csv", encoding="utf_8_sig", index=False, mode='w')

        if not os.path.exists(PATH_left + "\\" + i + "\\" + "shp"):
            os.makedirs(PATH_left + "\\" + i + "\\" + "shp")

        ogr.UseExceptions()


        shp_path = PATH_left + "\\" + i + "\\" + "shp"
        csv_path = PATH_left + "\\" + i + "\\" + "csv"

    for csv_filename in glob.glob(os.path.join(csv_path, '*.csv')):

        csv_df = pd.read_csv(csv_filename)


        driver = ogr.GetDriverByName('ESRI Shapefile')


        shp_filename = os.path.basename(csv_filename)[:-7] + '.shp'

        if os.path.exists(os.path.join(shp_path, shp_filename)):
            driver.DeleteDataSource(os.path.join(shp_path, shp_filename))
        ds = driver.CreateDataSource(os.path.join(shp_path, shp_filename))

        layer_name = os.path.basename(csv_filename)[:-7]

        sr = osr.SpatialReference()
        # 使用WGS84地理坐标系
        sr.ImportFromEPSG(4326)


        out_lyr = ds.CreateLayer(layer_name, srs=sr, geom_type=ogr.wkbPoint)


        lon_fld = ogr.FieldDefn('lon', ogr.OFTReal)
        lon_fld.SetWidth(50)
        lon_fld.SetPrecision(15)
        out_lyr.CreateField(lon_fld)
        # Latitude
        lat_fld = ogr.FieldDefn('lat', ogr.OFTReal)
        lat_fld.SetWidth(50)
        lat_fld.SetPrecision(15)
        out_lyr.CreateField(lat_fld)
        # ele
        ele_fld = ogr.FieldDefn('ele', ogr.OFTReal)
        ele_fld.SetWidth(50)
        ele_fld.SetPrecision(10)
        out_lyr.CreateField(ele_fld)
        # delta_time
        time_fld = ogr.FieldDefn('time', ogr.OFTString)
        time_fld.SetWidth(50)
        out_lyr.CreateField(time_fld)
        # quality_summary06
        quality_fld = ogr.FieldDefn('quality', ogr.OFTInteger)
        quality_fld.SetWidth(50)
        out_lyr.CreateField(quality_fld)

        featureDefn = out_lyr.GetLayerDefn()
        feature = ogr.Feature(featureDefn)

        point = ogr.Geometry(ogr.wkbPoint)


        for i in range(len(csv_df)):

            # ICESat-2

            feature.SetField('lon', float(csv_df.iloc[i, 0]))
            feature.SetField('lat', float(csv_df.iloc[i, 1]))
            feature.SetField('ele', float(csv_df.iloc[i, 2]))
            feature.SetField('time', str(csv_df.iloc[i, 3]))
            feature.SetField('quality', float(csv_df.iloc[i, 4]))


            point.AddPoint(float(csv_df.iloc[i, 0]), float(csv_df.iloc[i, 1]))
            feature.SetGeometry(point)


            out_lyr.CreateFeature(feature)

        ds.Destroy()

print('H5 data extraction and conversion completed')

'''

data clipping

'''

# 裁剪
guidao = {'gt1l', 'gt2l', 'gt3l'}

for i in guidao:

    if not os.path.exists(PATH_left + "\\" + i + "\\" + "shp_clip"):
        os.makedirs(PATH_left + "\\" + i + "\\" + "shp_clip")

    SHP_CLIP = PATH_left + "\\" + i + "\\" + "shp_clip"

    input_folder = PATH_left + "\\" + i + "\\" + "shp"
    clip_shapefile = "D:\青藏高原\青藏高原边界数据总集\TPBoundary_new(2021)\TPBoundary_new(2021).shp"  # Path to shp file for cropping
    output_folder = SHP_CLIP

    clip_shapefiles(input_folder, clip_shapefile, output_folder)

guidao = {'gt1r', 'gt2r', 'gt3r'}

for i in guidao:

    if not os.path.exists(PATH_right + "\\" + i + "\\" + "shp_clip"):
        os.makedirs(PATH_right + "\\" + i + "\\" + "shp_clip")

    SHP_CLIP = PATH_right + "\\" + i + "\\" + "shp_clip"


    input_folder = PATH_right + "\\" + i + "\\" + "shp"
    clip_shapefile = "D:\青藏高原\青藏高原边界数据总集\TPBoundary_new(2021)\TPBoundary_new(2021).shp"  # Path to shp file for cropping
    output_folder = SHP_CLIP


    clip_shapefiles(input_folder, clip_shapefile, output_folder)

print('Data cropping is completed')

'''

Merge cropped shp files

'''



def merge_shapefiles_in_folders(folders, output_file):

    merged_gdf = gpd.GeoDataFrame()


    for folder in folders:

        for filename in os.listdir(folder):
            if filename.endswith(".shp"):

                shp_path = os.path.join(folder, filename)
                gdf = gpd.read_file(shp_path)


                merged_gdf = gpd.GeoDataFrame(pd.concat([merged_gdf, gdf], ignore_index=True))


    merged_gdf.to_file(output_file, driver="ESRI Shapefile")



folders = [
    source_folder + '\\' + "+x\gt1r\shp_clip",
    source_folder + '\\' + "+x\gt2r\shp_clip",
    source_folder + '\\' + "+x\gt3r\shp_clip",
    source_folder + '\\' + "-x\gt1l\shp_clip",
    source_folder + '\\' + "-x\gt2l\shp_clip",
    source_folder + '\\' + "-x\gt3l\shp_clip"
]


output_file = source_folder + '\\' + "ICESat-2_clip.shp"


merge_shapefiles_in_folders(folders, output_file)

'''
shp to csv
'''
if not os.path.exists(source_folder + '\\' + "ICESat-2_clip"):
    os.makedirs(source_folder + '\\' + "ICESat-2_clip")


shp_file = output_file
gdf = gpd.read_file(shp_file)


csv_file = source_folder + '\\' + "ICESat-2_clip" '\\' + "ICESat-2_clip1.csv"
gdf.to_csv(csv_file, index=False)


csv_file1 = csv_file
df = pd.read_csv(csv_file1)


df = df.iloc[:, :-6]


output_file = source_folder + '\\' + "ICESat-2_clip" '\\' + "ICESat-2_clip.csv"
df.to_csv(output_file, index=False)

os.remove(csv_file)


print('Data consolidation completed')

'''
Zoning according to 1��
'''

source_path1 = source_folder + '\\' + "ICESat-2_clip"

output_folder1 = source_folder + '\\' + "1° partition of ICESat-2 data"


def process_and_save(file_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = os.listdir(file_path)
    for i in file_list:
        with open(os.path.join(file_path, i), "r", encoding="utf-8") as csv_file:
            data = csv_file.readlines()

        for line in data[1:]:
            line = line.strip()
            lon = int(line.split(',')[0].split('.')[0])
            lat = int(line.split(',')[1].split('.')[0])
            with open(os.path.join(output_folder, f"{lon}_{lat}.txt"), "a", encoding="utf-8") as f:
                f.write(line + '\n')



process_and_save(source_path1, output_folder1)

'''
Zoning according to 0.25°
'''

a = str(0.5)
b = a[1] + a[2]

Path_1d = source_folder + '\\' + "1° partition of ICESat-2 data"

if not os.path.exists(source_folder + '\\' + "0.25° partition of ICESat-2 data"):
    os.makedirs(source_folder + '\\' + "0.25° partition of ICESat-2 data")

out_path = source_folder + '\\' + "0.25° partition of ICESat-2 data"


def delete(file_path):
    file_list = os.listdir(file_path)
    for i in file_list:
        with open(file_path + '\\' + i, "r", encoding="utf-8") as csv_file:
            data = csv_file.readlines()

        for line in data:
            line = line.strip()
            lon = line.split(',')[0].split('.')[0]
            lat = line.split(',')[1].split('.')[0]
            lonxiaoshu = (float((line.split(',')[0].split('.')[1])[0] + (line.split(',')[0].split('.')[1])[1])) * 0.01
            latxiaoshu = (float((line.split(',')[1].split('.')[1])[0] + (line.split(',')[1].split('.')[1])[1])) * 0.01

            if float((line.split(',')[0].split('.')[1])[0]) < 5 and float(
                    (line.split(',')[1].split('.')[1])[0]) < 5:
                if lonxiaoshu < 0.25 and latxiaoshu < 0.25:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + "first_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu < 0.25:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + "second_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu < 0.25 and latxiaoshu >= 0.25:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + "third_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu >= 0.25:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + "fourth_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
            elif float((line.split(',')[0].split('.')[1])[0]) >= 5 and float(
                    (line.split(',')[1].split('.')[1])[0]) < 5:
                if lonxiaoshu < 0.75 and latxiaoshu < 0.25:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + "first_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.75 and latxiaoshu < 0.25:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + "second_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu < 0.75 and latxiaoshu >= 0.25:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + "third_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.75 and latxiaoshu >= 0.25:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + "fourth_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
            elif float((line.split(',')[0].split('.')[1])[0]) < 5 and float(
                    (line.split(',')[1].split('.')[1])[0]) >= 5:
                if lonxiaoshu < 0.25 and latxiaoshu < 0.75:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + str(b) + "first_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu < 0.75:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + str(b) + "second_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu < 0.25 and latxiaoshu >= 0.75:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + str(b) + "third_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu >= 0.75:
                    with open(out_path + '\\' + str(lon) + '_' + str(lat) + str(b) + "fourth_quadrant" + '.txt', "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
            elif float((line.split(',')[0].split('.')[1])[0]) >= 5 and float(
                    (line.split(',')[1].split('.')[1])[0]) >= 5:
                if lonxiaoshu < 0.25 and latxiaoshu < 0.75:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + str(b) + "first_quadrant" + '.txt',
                              "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu < 0.75:
                    with open(
                            out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + str(b) + "second_quadrant" + '.txt',
                            "a",
                            encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu < 0.25 and latxiaoshu >= 0.75:
                    with open(out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + str(b) + "third_quadrant" + '.txt',
                              "a",
                              encoding="utf-8") as f:
                        f.write(line + '\n')
                elif lonxiaoshu >= 0.25 and latxiaoshu >= 0.75:
                    with open(
                            out_path + '\\' + str(lon) + str(b) + '_' + str(lat) + str(b) + "fourth_quadrant" + '.txt',
                            "a",
                            encoding="utf-8") as f:
                        f.write(line + '\n')


delete(Path_1d)

'''
txt to csv
'''
txt_path1 = out_path


txt_path_name1 = os.listdir(txt_path1)


for file_name in txt_path_name1:
    if file_name.endswith(".txt"):

        txt_file_path = os.path.join(txt_path1, file_name)

        csv_file_path = os.path.join(txt_path1, os.path.splitext(file_name)[0] + ".csv")

        os.rename(txt_file_path, csv_file_path)

'''
Adding Column Names
'''

csv_add_listings = out_path


custom_column_names = ["lon", "lat", "H", "time"]


for filename in os.listdir(csv_add_listings):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_add_listings, filename)


        df = pd.read_csv(csv_path, header=None, names=custom_column_names)


        df.to_csv(csv_path, index=False)

print('0.25° partitioning of data completed')

'''
Extract crossover points
'''

EARTH_REDIUS = 6378.137

@jit
def rad(d):
    return d * math.pi / 180.0

@jit
def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS * 1000
    return s

@jit
def xulie3( x, y , H, TIME):
    l1 = len(x)
    result = []
    for s in range(l1):
        for k in range(s+1,l1):
            ds1 = getDistance(x[s],y[s],x[k],y[k])
            if ds1 <= 2:
                result.append([y[s], x[s], H[s], TIME[s],ds1])#
                result.append([y[k], x[k], H[k], TIME[k],ds1])#
            else:
                continue

    return result

def crossover_point(file_path):
    if not os.path.exists(file_path + '\\' + 'Crossover points without dh'):
        os.makedirs(file_path + '\\' + 'Crossover points without dh')
    out_file = file_path + '\\' + 'Crossover points without dh'
    file_list = os.listdir(file_path +'\\' + '0.25° partition of ICESat-2 data')
    for i in file_list:
        m = i[:-4]
        outpath = out_file + "\\" + m + "cjh" + "1" + ".txt"
        outpath1 = out_file + "\\" + m + "cjh" + "2" + ".txt"
        outpath2 = out_file + "\\" + m + "cjh" + "3" + ".txt"
        outpath3 = out_file + "\\" + m + "cjh" + "4" + ".txt"
        outpath4 = out_file + "\\" + m + "cjh" + "5" + ".txt"
        outpath5 = out_file + "\\" + m + "cjh" + "6" + ".txt"
        outpath6 = out_file + "\\" + m + "cjh" + "7" + ".txt"
        outpath7 = out_file + "\\" + m + "cjh" + "8" + ".txt"
        outpath8 = out_file + "\\" + m + "cjh" +  ".txt"

        icesat_2 = pd.read_csv(file_path +'\\' + '0.25° partition of ICESat-2 data' + "\\" + i ,encoding='utf-8',na_filter=False)

        lat1 = np.array(icesat_2["lat"])
        lon1 = np.array(icesat_2["lon"])
        h1 = np.array(icesat_2["H"])
        time1 = np.array(icesat_2["time"])

        p = xulie3(x=lat1 ,y=lon1,H=h1,TIME=time1,)#
        print(m)

        if not p:
            continue


        np.savez("p", p=np.array(p))
        file_handle=open(outpath,mode='w')
        file_handle.write(str(p))
        file_handle.close()


        import re
        f=open(outpath,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath1,'w+')
        for eachline in alllines:
            a=re.sub('],','\n',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath1,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath2,'w+')
        for eachline in alllines:
            a=re.sub(']]','',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath2,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath3,'w+')
        for eachline in alllines:
            a=re.sub("\[\[",'',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath3,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath4,'w+')
        for eachline in alllines:
            a=re.sub(" \[",'',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath4,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath5,'w+')
        for eachline in alllines:
            a=re.sub("\"\'",'',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath5,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath6,'w+')
        for eachline in alllines:
            a=re.sub("\'\"",'',eachline)
            f.writelines(a)
        f.close()

        f=open(outpath6,'r')
        alllines=f.readlines()
        f.close()
        f=open(outpath7,'w+')
        for eachline in alllines:
            a=re.sub(" ",'',eachline)
            f.writelines(a)
        f.close()

        f = open(outpath7, 'r')
        alllines = f.readlines()
        f.close()
        f = open(outpath8, 'w+')
        for eachline in alllines:
            a = re.sub("'", '', eachline)
            f.writelines(a)
        f.close()

        os.remove(outpath)
        os.remove(outpath1)
        os.remove(outpath2)
        os.remove(outpath3)
        os.remove(outpath4)
        os.remove(outpath5)
        os.remove(outpath6)
        os.remove(outpath7)

crossover_point(source_folder)


txt_path = source_folder + '\\' + 'Crossover points without dh'

txt_path_name = os.listdir(txt_path)


for file_name in txt_path_name:
    if file_name.endswith(".txt"):

        txt_file_path = os.path.join(txt_path, file_name)

        csv_file_path = os.path.join(txt_path, os.path.splitext(file_name)[0] + ".csv")

        os.rename(txt_file_path, csv_file_path)

Folder_Path = source_folder + '\\' + 'Crossover points without dh'
if not os.path.exists(source_folder + '\\' + 'Crossover points without dh'):
    os.makedirs(source_folder + '\\' + 'Crossover points without dh')
SaveFile_Path = source_folder + '\\' + 'Crossover points without dh'
SaveFile_Name = r'Crossover points without dh.csv'


os.chdir(Folder_Path)

file_list = os.listdir()


df = pd.read_csv(Folder_Path + '\\' + file_list[0], header=0)


df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, line_terminator='\n')


for i in range(1, len(file_list)):
    df = pd.read_csv(Folder_Path + '\\' + file_list[i], header=None)
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+',
              line_terminator='\n')

print('The synthesized crossover points file is：Crossover points without dh.csv')


"""
Calculate dh
"""

if not os.path.exists(source_folder + '\\' + 'Final crossover points'):
    os.makedirs(source_folder + '\\' + 'Final crossover points')

Crossover_point = source_folder + '\\' + 'Crossover points without dh' + '\\' + 'Crossover points without dh.csv'
Odd = source_folder + '\\' + 'Final crossover points' + '\\' + 'Odd.csv'
Even = source_folder + '\\' + 'Final crossover points' + '\\' + 'Even.csv'
Crossover_point_with_dh = source_folder + '\\' + 'Final crossover points' + '\\' + 'Crossover_point_with_dh.txt'

with open(Crossover_point, 'r', encoding='utf-8-sig') as infile, \
        open(Odd, 'w', newline='', encoding='utf-8-sig') as oddfile, \
        open(Even, 'w', newline='', encoding='utf-8-sig') as evenfile:

    lines = infile.readlines()


    fieldnames = ['lon', 'lat', 'H', 'time', 'ds']
    odd_writer = csv.DictWriter(oddfile, fieldnames=fieldnames)
    even_writer = csv.DictWriter(evenfile, fieldnames=fieldnames)


    odd_writer.writeheader()
    even_writer.writeheader()


    for i in range(0, len(lines), 2):
        values_odd = lines[i].strip().split(',')
        values_even = lines[i + 1].strip().split(',')

        row_data_odd = {
            'lon': values_odd[0],
            'lat': values_odd[1],
            'H': values_odd[2],
            'time': values_odd[3],
            'ds': values_odd[4]
        }

        row_data_even = {
            'lon': values_even[0],
            'lat': values_even[1],
            'H': values_even[2],
            'time': values_even[3],
            'ds': values_even[4]
        }


        odd_writer.writerow(row_data_odd)

        even_writer.writerow(row_data_even)



def demo(day1, day2):
    time_array1 = time.strptime(day1, "%Y-%m-%d")
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(day2, "%Y-%m-%d")
    timestamp_day2 = int(time.mktime(time_array2))
    result = (timestamp_day1 - timestamp_day2) // 60 // 60 // 24
    if result > 0:
        return True
    else:
        return False


def timeorder(x1, x2, y1, y2, H1, H2, TIME1, TIME2, ds1, ds2):
    l1 = len(TIME1)
    result = []
    for i in range(l1):
        if demo(TIME1[i], TIME2[i]) == True:
            dh = H1[i] - H2[i]
            result.append([i, x2[i], y2[i], H2[i], TIME2[i], ds2[i], dh])
            result.append([i, x1[i], y1[i], H1[i], TIME1[i], ds1[i], dh])
        else:
            dh = H2[i] - H1[i]
            result.append([i, x1[i], y1[i], H1[i], TIME1[i], ds1[i], dh])
            result.append([i, x2[i], y2[i], H2[i], TIME2[i], ds2[i], dh])
    return result


file1 = pd.read_csv(Odd, encoding='utf-8')
file2 = pd.read_csv(Even, encoding='utf-8')

lon1 = np.array(file1["lon"])
lon2 = np.array(file2["lon"])
lat1 = np.array(file1["lat"])
lat2 = np.array(file2["lat"])
h1 = np.array(file1["H"])
h2 = np.array(file2["H"])
time1 = np.array(file1["time"])
time2 = np.array(file2["time"])
ds1 = np.array(file1["ds"])
ds2 = np.array(file2["ds"])

p = timeorder(x1=lon1, x2=lon2, y1=lat1, y2=lat2, H1=h1, H2=h2, TIME1=time1, TIME2=time2, ds1=ds1, ds2=ds2)
np.savez("p", p=np.array(p))
with open(Crossover_point_with_dh, 'w', encoding='utf-8') as f:
    print(p, file=f)

f = open(Crossover_point_with_dh, 'r', encoding='utf-8')
alllines = f.readlines()
f.close()
f = open(Crossover_point_with_dh, 'w+', encoding='utf-8')
for eachline in alllines:
    a = re.sub('],', '\n', eachline)
    f.writelines(a)
f.close()

f = open(Crossover_point_with_dh, 'r', encoding='utf-8')
alllines = f.readlines()
f.close()
f = open(Crossover_point_with_dh, 'w+', encoding='utf-8')
for eachline in alllines:
    a = re.sub('\'', '', eachline)
    f.writelines(a)
f.close()

f = open(Crossover_point_with_dh, 'r', encoding='utf-8')
alllines = f.readlines()
f.close()
f = open(Crossover_point_with_dh, 'w+', encoding='utf-8')
for eachline in alllines:
    a = re.sub(' ', '', eachline)
    f.writelines(a)
f.close()

f = open(Crossover_point_with_dh, 'r', encoding='utf-8')
alllines = f.readlines()
f.close()
f = open(Crossover_point_with_dh, 'w+', encoding='utf-8')
for eachline in alllines:
    a = re.sub('\[', '', eachline)
    f.writelines(a)
f.close()

f = open(Crossover_point_with_dh, 'r', encoding='utf-8')
alllines = f.readlines()
f.close()
f = open(Crossover_point_with_dh, 'w+', encoding='utf-8')
for eachline in alllines:
    a = re.sub(']]', '', eachline)
    f.writelines(a)
f.close()

os.remove(Even)
os.remove(Odd)


def change_file_extension(filename, new_extension):

    name, extension = os.path.splitext(filename)

    new_filename = name + new_extension

    os.rename(filename, new_filename)




change_file_extension(Crossover_point_with_dh, '.csv')

df = pd.read_csv(source_folder + '\\' + 'Final crossover points' + '\\' + 'Crossover_point_with_dh.csv', header=None,
                 names=['number', 'lon', 'lat', 'H', 'time', 'ds', 'dh'])
df.to_csv(source_folder + '\\' + 'Final crossover points' + '\\' + 'Final_Crossover_points(Outliers_not_removed).csv', encoding="utf_8_sig", index=False,
          mode='w')

os.remove(source_folder + '\\' + 'Final crossover points' + '\\' + 'Crossover_point_with_dh.csv')


'''
Removal of outliers
'''
with open(source_folder + '\\' + 'Final crossover points' + '\\' + 'Final_Crossover_points(Outliers_not_removed).csv') as csv_file:
    row = csv.reader(csv_file, delimiter=',')

    next(row)
    price = []


    for r in row:
        price.append(float(r[6]))

print('RMSE (before removing outliers) =', math.sqrt(np.var(price)))
print('MEAN (before removing outliers) =',np.mean(price))
max = np.mean(price) + 3 * math.sqrt(np.var(price))
min = np.mean(price) - 3 * math.sqrt(np.var(price))

input_file = source_folder + '\\' + 'Final crossover points' + '\\' + 'Final_Crossover_points(Outliers_not_removed).csv'
output_file = source_folder + '\\' + 'Final crossover points' + '\\' + 'Final_Crossover_points.csv'

important_dates = [0, 10000]
with open(input_file, 'r', newline='') as csv_in_file:
    with open(output_file, 'w', newline='') as csv_out_file:
        filereader = csv.reader(csv_in_file)
        filewriter = csv.writer(csv_out_file)
        header = next(filereader)
        filewriter.writerow(header)
        for row_list in filereader:
            cost = row_list[6]
            if  float(cost) > min and float(cost) < max:
                filewriter.writerow(row_list)


with open(source_folder + '\\' + 'Final crossover points' + '\\' + 'Final_Crossover_points.csv') as csv_file:
    row = csv.reader(csv_file, delimiter=',')

    next(row)
    price = []


    for r in row:
        price.append(float(r[6]))

print('RMSE (after removing outliers) =', math.sqrt(np.var(price)))
print('MEAN (after removing outliers) =',np.mean(price))

print('The final ICESat-2 crossover points file is：Final_Crossover_points.csv')
