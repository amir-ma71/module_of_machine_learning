from tkinter.filedialog import askopenfilename
import pandas as pd

# load Data
def load_data():
    # select separator
    separator = input("لطفا جداکننده را وارد کرده و سپس فایل دیتاست را انتخاب کنید: \n")
    # reading file path
    filepath = askopenfilename()
    dataset = pd.read_csv(filepath, sep=separator, encoding="utf-8")
    featurelist = list(dataset)

    # delete features that don't use
    print("لیست ویژگی های دیتاست به شرح ذیل است:")
    for i in range(len(featurelist)):
        print(i, featurelist[i])
    deleteAtr = input("آیا تمایل به حذف تعدادی از ویژگی ها را دارید؟)Y/N) \n")
    delete_Atr = []
    if deleteAtr == "Y" or deleteAtr == "y":
        print(
            "لطفا اندیس ویژگی ای که قصد حذف آن را دارید وارد کرده و بعد از هر اندیس اینتر بزنید و در انتها * را وارد کنید")
        while True:
            c = input()
            if c == "*":
                break
            else:
                delete_Atr.append(featurelist[int(c)])
    dataset.drop(delete_Atr, axis=1, inplace=True)

    # select Label and put it at the end of list
    label = featurelist[int(input("لطفا اندیس برچسب را وارد کنید\n"))]
    label2 = dataset.pop(label)
    dataset[label] = label2

    return dataset