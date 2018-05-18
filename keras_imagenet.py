from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import glob

model = ResNet50(weights='imagenet')
model.summary()
##########################################################
# parameters

topn = 2  # number of top guesses to consider


##########################################################
# folder names
labels = ['background','braceroot','foreignobject','ground','leaf','stem']

''' track what the network guesses for each label
first guess, second guess etc are stored in seperate dictionaries
dic_first_guess = {
    label1: [num_guesses, [confidence1,confidence2,...confidencen]]
    label2: [num_guesses, [confidence1,confidence2]]
}'''

results = []

for index, label in enumerate(labels):
    # loop for each label 

    print('reading {} files (Index: {})'.format(label,index))
    path = os.path.dirname(__file__) + '/clean-dataset/train/' + label

    # create list of dictionaries
    pre_dict = []
    for n in range(topn):
        prediction = {}
        pre_dict.append(prediction)

    for filepath in glob.glob(os.path.join(path, '*png')):
        # loop for each photo in a label's folder

        # feed image through net and store predictions in 'predictions'
        img_path = filepath
        img = image.load_img(img_path, target_size=(224, 224))
        preds = model.predict(preprocess_input(
            x=np.expand_dims(image.img_to_array(img), axis=0)))
        predictions = decode_predictions(preds, top=topn)[0]

        # record prediction information into dictionary
        for index, p in enumerate(predictions):
            # loops through first guess, second guess etc..
            label = p[1]
            # add 1 to and store confidence in label 'p[1]'
            try:
                pre_dict[index][label][0] = pre_dict[index][label][0] + 1
                pre_dict[index][label][1].append(p[2])
            except KeyError:
                pre_dict[index][label] = [1,[p[2]]]
    results.append(pre_dict)

''' 
Creates a dictionary for each groundtruth label containing the prediction information
Essentially just removes information about the guess order (first guess, second guess)
'''
all_guesses = []
for label in results:
    guesses = {}
    for guess_n in label:
        for key,value in guess_n.items():
            try:
                guesses[key][0] = guesses[key][0] + value[0]
                guesses[key][1] = guesses[key][1] + value[1]
            except KeyError:
                guesses[key] = value
    all_guesses.append(guesses)

''' 
Turns dictionary into a list and also calculates the mean and std of the confidences
'''
outputs = []
for label in all_guesses:
    label_output = []
    for key,value in label.items():
        name = key
        occurances = value[0]
        mean = np.mean(value[1])
        standard_deviation = np.std(value[1])
        output = [name,occurances,mean,standard_deviation]
        label_output.append(output)
    outputs.append(label_output)

''' 
Print results
'''
print('\n','\n','\n')
for label_output,label in zip(outputs,labels):
    print('guesses for {}'.format(label) + '\n')
    for output in label_output:
        print('name: '+str(output[0]).ljust(30),' occurances: '+str(output[1]).ljust(5),' average confidence: '+"{0:.3f}".format(output[2]) .ljust(7),' confidence std: '+"{0:.3f}".format(output[3]))
    print('\n','\n')
