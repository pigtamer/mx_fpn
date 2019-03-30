# cunyuan

import json,os,subprocess,time
import argparse
import cv2 as cv

if not os.path.exists('output'):
    os.makedirs('output')

def write_xml(bndbox,size, lbl, filename):
    width, height = size
    xmin, ymin, xmax, ymax = bndbox
    with open(filename, 'w') as f:
        f.writelines("<annotation verified=\"yes\">\n")


        f.writelines("<size>\n")

        f.writelines("<width>%s</width>\n"%width)
        f.writelines("<height>%s</height>\n"%height)

        f.writelines("</size>\n")


        f.writelines("<object>\n")
        f.writelines("<name>%s</name>\n"%lbl)
        f.writelines("<bndbox>\n")

        f.writelines("<xmin>%s</xmin>\n"%xmin)
        f.writelines("<ymin>%s</ymin>\n"%ymin)
        f.writelines("<xmax>%s</xmax>\n"%xmax)
        f.writelines("<ymax>%s</ymax>\n"%ymax)

        f.writelines("</bndbox>\n")
        f.writelines("</object>\n")


def labelj2xml():
    with open('img.json', 'r') as f:
        data_dict = json.load(f)
        idx = 0
        for frame_name in data_dict["frames"]:
            if data_dict["frames"]["%s"%frame_name]!=[]:
                each_frame = cv.imread("./img/%s"%frame_name)
                cv.imwrite("./output/%s"%frame_name, each_frame)
                xmin = (data_dict["frames"]["%s"%frame_name][0]["x1"])
                ymin = (data_dict["frames"]["%s"%frame_name][0]["y1"])
                xmax = (data_dict["frames"]["%s"%frame_name][0]["x2"])
                ymax = (data_dict["frames"]["%s"%frame_name][0]["y2"])
                lbl = (data_dict["frames"]["%s"%frame_name][0]["tags"][0])
                width= (data_dict["frames"]["%s"%frame_name][0]["width"])
                height = (data_dict["frames"]["%s"%frame_name][0]["height"])
                write_xml((xmin,ymin, xmax, ymax), (width, height), lbl, "./output/%s.xml"%frame_name[0:-4])
                idx += 1
            else:
                pass
                # print("==\n==\n"*5)
        f.close()
    print("%s labeled images processed."%idx)


def labelj2lst(data_path):
    lst = []
    fl = open(data_path+'train.lst','w')
    with open('img.json', 'r') as f:
        data_dict = json.load(f)
        idx = 0
        for frame_name in data_dict["frames"]:
            if data_dict["frames"]["%s"%frame_name]!=[]:
                each_frame = cv.imread("./img/%s"%frame_name)
                cv.imwrite(data_path + "%s"%frame_name, each_frame)
                xmin = (data_dict["frames"]["%s"%frame_name][0]["x1"])
                ymin = (data_dict["frames"]["%s"%frame_name][0]["y1"])
                xmax = (data_dict["frames"]["%s"%frame_name][0]["x2"])
                ymax = (data_dict["frames"]["%s"%frame_name][0]["y2"])
                lbl = (data_dict["frames"]["%s"%frame_name][0]["tags"][0])
                width= (data_dict["frames"]["%s"%frame_name][0]["width"])
                height = (data_dict["frames"]["%s"%frame_name][0]["height"])
                img_name = data_path + frame_name

                lst_tmp =str(idx)+'\t4'+'\t5'+'\t'+str(width)+'\t'+str(height)+'\t'\
                +str('1')+'\t'\
                +str(xmin/width)+'\t'+str(ymin/height)+'\t'\
                +str(xmax/width)+'\t'+str(ymax/height)+'\t'\
                + img_name +'\n'
                #print(lst_tmp)
                fl.write(lst_tmp)

                idx += 1
            else:
                pass
                # print("==\n==\n"*5)
            # if idx==1000: break
    f.close()
    fl.close()

    print("%s labeled images processed."%idx)


labelj2lst("output/")
os.system("python3 im2rec.py output/train.lst ./ --pack-label")