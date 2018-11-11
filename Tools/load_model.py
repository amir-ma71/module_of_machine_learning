
# ask user for choose the model that need
# to be continue to complete...

def load_model():
    num_model = int(input("لطفا عدد مدل را انتخاب کنید: \n 1. درخت تصمیم \n 2. شبکه عصبی\n"))
    if num_model == 1:
        num_algh = int(input("لطفا نوع درخت تصمیم را انتخاب کنید: \n 1. طبقه بندی \n 2. رگرسیون\n"))
        if num_algh == 1:
            num_algh2 = int(input("لطفا نوع الگوریتم را انتخاب کنید: \n 1. جینی \n 2. انتروپی\n"))
            DecisionTree.learn_model(load_data(),num_algh2)
        else:
            DecisionTree.learn_model(load_data(),3)
    return

