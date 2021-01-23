from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import shutil
import pandas as pd
import pickle
from tqdm import tqdm


# r,g,b degerleri verilen pixel icin hue degeri hesaplama

def get_hue(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    elif mx == b:
        h = (60 * ((r - g) / df) + 240) % 360
    return int(h)


# Histogramlari hesaplayan fonksiyon.
# dosya_listesi: histogramlari hesaplanacak fotograflarin path'leri
# filename: .pkl olarak kaydetmek icin verilen ad
# force_new_create: True icin sifirdan histogram hesabi, False icin '.pkl' uzantili dosyadan histogramlari okuma

def get_histograms(dosya_listesi, filename, force_new_create, save):
    if os.path.exists(filename) and force_new_create is False:
        df = pd.read_pickle(filename)
    else:
        df = pd.DataFrame()

        # tqdm: histogramlari hesaplarken gidisatin gozukmesi icin kullanilan basit bir kutuphane

        pbar = tqdm(range(0, len(dosya_listesi)), desc='Creating histograms')

        for index in pbar:
            # fotograflar tek tek okunuyor
            x = dosya_listesi[index]

            # opencv'nin pathi dogru almasi icin gerekli preprocess
            x_path = x.split('/')[-3] + '/' + x.split('/')[-2] + '/' + x.split('/')[-1]
            img = cv2.imread(x_path)

            row, col = img.shape[0], img.shape[1]

            hue_list = np.zeros((360,))
            r_list = np.zeros((256,))
            g_list = np.zeros((256,))
            b_list = np.zeros((256,))

            # fotograftaki pixeller icin hue degeri hesaplama
            for i in range(col):
                for j in range(row):
                    h = get_hue(img[j, i, 2], img[j, i, 1], img[j, i, 0])
                    hue_list[h] += 1

                    r = img[j, i, 2]
                    r_list[r] += 1

                    g = img[j, i, 1]
                    g_list[g] += 1

                    b = img[j, i, 0]
                    b_list[b] += 1

            # r,g,b ve hue listeleri icin normalizasyon
            r_list /= col * row
            g_list /= col * row
            b_list /= col * row
            hue_list /= col * row

            # histogrami hesaplanan fotografin bilgilerinin kaydedilmesi
            instance = {'dosya_adi': x.split('/')[-1], 'class': x.split('/')[-2], 'hue_list': hue_list,
                        'r_list': r_list, 'g_list': g_list, 'b_list': b_list}
            df = df.append(instance, ignore_index=True)

        # dataframe olarak tutulan bilgileri .pkl uzantili dosya olarak local'e kaydetme
        # yapilmasi gerekli degil fakat ayni train ve test fotograflarini farkli zamanlarda test icin kolaylik sagliyor.
        if save:
            df.to_pickle(filename)

    return df


# 2 deger icin oklid mesafesi hesaplama

def euclid_dist(val1, val2):
    dist = math.sqrt((val1 - val2) ** 2)
    return dist


# verilen hue, r,g,b histogramlari icin hue ve rgb oklid mesafeleri hesaplama

def euclid(img1_hue, img1_r, img1_g, img1_b, img2_hue, img2_r, img2_g, img2_b):
    hue_dist, r_dist, g_dist, b_dist = 0, 0, 0, 0

    for x in range(len(img1_hue)):
        hue_dist = hue_dist + euclid_dist(img1_hue[x], img2_hue[x])

    for x in range(len(img1_r)):
        r_dist = r_dist + euclid_dist(img1_r[x], img2_r[x])
        g_dist = g_dist + euclid_dist(img1_g[x], img2_g[x])
        b_dist = b_dist + euclid_dist(img1_b[x], img2_b[x])

    rgb_dist = math.sqrt(r_dist ** 2 + g_dist ** 2 + b_dist ** 2)

    return hue_dist, rgb_dist


# rastgele 30 dosya secme ve bunlari 25 train, 5 test olarak ayirma

def train_test_split(dosya_paths):
    train_images = []
    test_images = []

    for dosya_path in dosya_paths:

        list_dir = os.listdir(dosya_path)
        random_numbers = random.sample(range(0, len(list_dir)), 30)

        train_list = random_numbers[0:25]
        test_list = random_numbers[25:]

        for i in train_list:
            train_images.append(dosya_path + list_dir[i])
        for i in test_list:
            test_images.append(dosya_path + list_dir[i])

    return train_images, test_images


# verilen train ve test histogram dataframeleri icin oklid hue ve rgb mesafeleri hesaplama
# hesaplanan mesafelerderden en yakin 5 fotografin test fotografiyla ayni olup olmadigini bulma
# ve basari sonuclarini paylasma

def test(train_df, test_df):
    print('Preparing to test..')
    results_df = pd.DataFrame()

    # her test fotografini her egitim fotografi ile karsilastirarak mesafelerini hesaplama

    for i in test_df.index:
        for j in train_df.index:
            hue_dist, rgb_dist = euclid(train_df['hue_list'][j], train_df['r_list'][j], train_df['g_list'][j],
                                        train_df['b_list'][j],
                                        test_df['hue_list'][i], test_df['r_list'][i], test_df['g_list'][i],
                                        test_df['b_list'][i])
            results_df = results_df.append({'train_name': train_df['dosya_adi'][j],
                                            'test_name': test_df['dosya_adi'][i], 'hue_dist': hue_dist,
                                            'rgb_dist': rgb_dist}, ignore_index=True)

    # test fotograflarinin listesi
    test_list = (results_df.test_name.unique())

    accuracy_results = []
    # hue ve rgb mesafelerine gore ayri ayri bakacagiz
    for col in ['hue_dist', 'rgb_dist']:
        print(f'RESULTS FOR {col}:')

        # her class a ait dogruluk orani tutulacak
        true_counts = {
            '028_': 0,
            '056_': 0,
            '057_': 0,
            '084_': 0,
            '089_': 0,
            '105_': 0
        }

        for test_image in test_list:
            # top_5_df: her test fotografi icin en yakin 5 sonucun bulundugu dataframe
            top_5_df = results_df[results_df['test_name'] == test_image].sort_values(by=col).head(5)

            # train_names: test fotografi icin en yakin 5 train fotografi listesi
            train_names = top_5_df['train_name'].values
            train_names = [x[:4] for x in train_names]

            # class codunun bulundugu ilk 4 karakter onemli, ornegin:028. ise Camel
            if test_image[:4] in train_names:
                # true_counts a % lik olarak dogruluk orani kaydediliyor.
                # 5 test fotografinin her birinin % lik agirligi: 100/ (30/6):
                # toplam 30 test ve 6 class var(her class dan 5 test)

                true_counts[test_image[:4]] = true_counts[test_image[:4]] + (100/(len(test_list)/len(true_counts)))

            print(f'{test_image} results:')
            print(top_5_df)
            print('------------------------')
            # break
        # toplam basari kaydedildi
        accuracy_results.append(true_counts)

    # toplam basari
    for i, col in enumerate(['hue_dist', 'rgb_dist']):
        accuracy_values = list(accuracy_results[i].values())
        print(f'CLASS ACCURACY of {col}: {accuracy_results[i]}')
        print(f'OVERALL ACCURACY of {col}: {sum(accuracy_values) / len(accuracy_values)}%')
        print('------------------------')

    return results_df


'''
dataset_path ile belirtilen klasorun icerisinde 
028.camel, 056.dog, 057.dolphin, 084.giraffe, 089.goose, 105.horse 
alt klasorleri bulunmalidir.
'''

dataset_path = 'C:/Users/thewa/Desktop/lecture/görüntü işleme/ödev2/dataset'

# icerdeki klasorlerin yolunu aliyoruz
file_paths = list(map(lambda x: dataset_path + '/' + x + '/', os.listdir(dataset_path)))

# her sinif icin 25 egitim, 5 test toplam 30 fotograf seciliyor
train_list, test_list = train_test_split(file_paths)

# force_new_create parametresi True icin histogramlari sifirdan hesaplar ve
# '.pkl' uzantili dosya olarak kaydeder.

# False icin, daha onceden ayrilan egitim ve test fotograflari icin
# hesaplanmis ve '.pkl' uzantili dosya olarak kaydedilmis histogramlari kullanir.

# save: True icin histogramlari hesaplar ve '.pkl' olarak kaydeder.
# False icin sadece hesaplar ve runtime suresince dataframe olarak bellekte tutar.

train_df = get_histograms(train_list, 'train.pkl', force_new_create=True, save=False)
test_df = get_histograms(test_list, 'test.pkl', force_new_create=True, save=False)

# histogramlar test ediliyor
results_df = test(train_df, test_df)
