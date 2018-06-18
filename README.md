# corn classifier 
I created a two simple models to evaluate images of various parts of corn plants

keras-imagenet.py - returns the output of a pre-trained and unaltered ResNet50 on the data
transfer.py - uses transfer learning from a pre-trained ResNet50 to train a model with labels from our data

## The data
### description
The data is screenshot from the videos on the ARPA_e folder on box. Since I was just screenshoting, I was able to collect images from 6 different videos. The categories are:

stem - exposed parts of the corn’s stem
leaf - leaves still attached to stalk
brace root - above-ground root structure seen growing from some stems
ground - dirt 
background - other corn plants in the background
foreign object - anything else (shoes, blackboard, white tube end)

The clean-dataset folder in this repository contains a few random examples of the  images for each of these classes. 

The full dataset consist of 178 training images and 67 evaluation images. I did the best I could to make sure the images in the evaluation dataset were taken from separate videos to improve the validity of the evaluation dataset. 

### comments 
While there are a limited number of labels, it is still difficult to classify for the following reasons:

1) Limited data - ~30 images per class is not many images to train on
2) All the data exists within the same context, which can create confusion. For example, an image of a leaf may have a stem in the background which leads to misclassification. 
3) Loss of image size information - ResNet50 takes 224x224 px images. If an image too small or large, it gets resized to fit the bounding box and the size information is lost. Since we are training a very domain-specific model, size may be one of the most significant pieces of information to distinguish between for example a leaf in the foreground and a leaf in the background. 

## Keras-imagenet.py
### description
A simple use of  a ResNet50 pre-trained on imagenet. Keras conveniently has this functionality built in, and I used some code from an example implementation [https://keras.io/applications/#resnet50]  which classifies an image of an elephant. 

I fed the training images through the network and observed its predictions.
For each ground truth label, it records each predicted label, the number of times it was predicted, and the mean and standard deviation of its confidence in each predicted label. 

### results
The results are abysmal, which is expected given that our categories don’t exist on the imagenet dataset. The full results are in (1) of the appendix at the bottom of this document. 

Though you mentioned this to me before, I was still surprised that ’ear’ was not only of the outputs, but also by far the most common prediction by the ResNet: 19 out of 37 predictions for ‘leaf’ and 15/49 for ‘stem’

Most of the predictions are of organisms found in nature. I’m not sure if this is more a reflection of the nature in my dataset or a result of the ResNet model (from the amount of nature categories on the imagenet dataset)

## transfer.py

## FIRST RUN, PROBLEM WITH CODE SO NOT SURE IF RESULTS ARE RELIABLE
### description
transfer.py implements transfer learning. Essentially, it takes a pre-trained ResNet50 model, and replaces the final fully connected and output layers with two new, untrained layers. The original model had a fully connected layer with 2048 neurons but I took it down to 512 neurons because we only need output neurons instead of 1000.  Only these last two layers are trained, since the idea is that the pre-trained network has already learned the relevant features. Training all 50 layers usually gives better results, but I did not have the computational power or time to do that. 

The code is based off of an implementation by [www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/]. I added comments and made it work with ResNet50 instead of VGG16 along with a few other tweaks including hyper parameter tuning. 

Details about the training can be found in the code and on the training log in (2) of the appendix. 

### results
Out of all the instances, the best results achieved was an evaluation accuracy of 47/67 on the training dataset. This is not a very good accuracy, but with more hyper-parameter tuning and much more data I believe this accuracy can go up drastically. Furthermore, ‘foreign objects’ is a questionable category, as their similarities lie only in their difference from the other categories. 

With more time, I would take a look at which images the errors are occurring for and how the network was wrong to evaluate the cause for the errors. 

It could just be the foreign object category with all the issues. Or, it could be different lighting conditions or different phases of the crop cycle between different videos, which could be improved with more data. Or, it could be confusion caused by objects in the background, which could mean a need for different bounding box method. 

## SECOND PART
### description
I tried applying transfer learning using a ResNet50 pretrained on ImageNet to classify 3 labels: [brown leaf, green leaf, stem] but haven’t been getting good results. The highest evaluation accuracy the model achieved was 48% accuracy (hardly better than a random 33%), which was achieved with a low learning rate and training until accuracy or loss began dropping due to overfitting.

Architecture:
[224x224] image > Pretrained ResNet50* > untrained FC layer > output (3 neurons)
*without output or final fully connected layer

### results
see transfer_results.png

### evaluation
Here are some possibilities for why it’s performing so poorly, since I can’t seem to find hyper-parameters to get eval_acc above 50%

   1. Needs more data
The training and evaluation set both contain only 250-300 images total. This is nowhere near the amount of data used by other recent DL models. It’s also worth noting that despite the training and evaluation data coming from separate videos, the lighting in those two videos is quite similar and doesn’t represent the expected training domain. Lack of data certainly is one of the issues, just not sure how much it contributes
   2. Transfer learning won’t work
The pretrained ResNet50 model likely contains heaps of irrelevant features, since ImageNet covers a broader domain than just images of corn. This could make training slow, noisy and generally difficult. Perhaps a model trained specifically for our domain is necessary
   3. Data needs improvement
Since training data is cropped from individual frames based on our annotations, training images don’t have consistent dimensions and are scaled to be 224x224, which inconsistently and drastically distorts images
   4. Misc 
I haven’t tried using other optimizers or changing the size of the untrained FC layer or how training data is sampled yet

### appendix

1)
```
guesses for background

name: capuchin                        occurances: 1      average confidence: 0.073    confidence std: 0.000
name: bittern                         occurances: 9      average confidence: 0.136    confidence std: 0.087
name: European_gallinule              occurances: 2      average confidence: 0.133    confidence std: 0.106
name: coucal                          occurances: 2      average confidence: 0.129    confidence std: 0.049
name: langur                          occurances: 1      average confidence: 0.064    confidence std: 0.000
name: coral_fungus                    occurances: 2      average confidence: 0.095    confidence std: 0.013
name: cardoon                         occurances: 7      average confidence: 0.186    confidence std: 0.157
name: proboscis_monkey                occurances: 2      average confidence: 0.228    confidence std: 0.108
name: pot                             occurances: 1      average confidence: 0.387    confidence std: 0.000
name: ear                             occurances: 5      average confidence: 0.334    confidence std: 0.256
name: whiptail                        occurances: 1      average confidence: 0.049    confidence std: 0.000
name: titi                            occurances: 3      average confidence: 0.118    confidence std: 0.077
name: squirrel_monkey                 occurances: 1      average confidence: 0.127    confidence std: 0.000
name: worm_fence                      occurances: 3      average confidence: 0.221    confidence std: 0.211
name: amphibian                       occurances: 1      average confidence: 0.176    confidence std: 0.000
name: walking_stick                   occurances: 1      average confidence: 0.051    confidence std: 0.000
name: little_blue_heron               occurances: 3      average confidence: 0.118    confidence std: 0.040
name: limpkin                         occurances: 1      average confidence: 0.067    confidence std: 0.000
name: corn                            occurances: 2      average confidence: 0.131    confidence std: 0.043
name: artichoke                       occurances: 3      average confidence: 0.152    confidence std: 0.088
name: pineapple                       occurances: 1      average confidence: 0.054    confidence std: 0.000
name: green_mamba                     occurances: 1      average confidence: 0.172    confidence std: 0.000
name: siamang                         occurances: 1      average confidence: 0.159    confidence std: 0.000
name: stinkhorn                       occurances: 2      average confidence: 0.070    confidence std: 0.001

 

guesses for braceroot

name: titi                            occurances: 2      average confidence: 0.108    confidence std: 0.052
name: vine_snake                      occurances: 1      average confidence: 0.108    confidence std: 0.000
name: cucumber                        occurances: 2      average confidence: 0.079    confidence std: 0.010
name: stinkhorn                       occurances: 3      average confidence: 0.054    confidence std: 0.014
name: sea_anemone                     occurances: 5      average confidence: 0.264    confidence std: 0.120
name: frilled_lizard                  occurances: 1      average confidence: 0.078    confidence std: 0.000
name: green_mamba                     occurances: 1      average confidence: 0.078    confidence std: 0.000
name: chambered_nautilus              occurances: 1      average confidence: 0.188    confidence std: 0.000
name: banana                          occurances: 1      average confidence: 0.277    confidence std: 0.000
name: ear                             occurances: 1      average confidence: 0.106    confidence std: 0.000
name: custard_apple                   occurances: 5      average confidence: 0.121    confidence std: 0.032
name: damselfly                       occurances: 1      average confidence: 0.143    confidence std: 0.000
name: capuchin                        occurances: 1      average confidence: 0.053    confidence std: 0.000
name: cardoon                         occurances: 1      average confidence: 0.087    confidence std: 0.000
name: hen-of-the-woods                occurances: 1      average confidence: 0.087    confidence std: 0.000
name: coral_fungus                    occurances: 1      average confidence: 0.030    confidence std: 0.000
name: howler_monkey                   occurances: 1      average confidence: 0.093    confidence std: 0.000
name: bittern                         occurances: 1      average confidence: 0.042    confidence std: 0.000
name: corn                            occurances: 1      average confidence: 0.066    confidence std: 0.000
name: agaric                          occurances: 1      average confidence: 0.261    confidence std: 0.000
name: earthstar                       occurances: 1      average confidence: 0.099    confidence std: 0.000
name: hoopskirt                       occurances: 1      average confidence: 0.091    confidence std: 0.000

 

guesses for foreignobject

name: mountain_tent                   occurances: 2      average confidence: 0.103    confidence std: 0.036
name: scabbard                        occurances: 1      average confidence: 0.150    confidence std: 0.000
name: bulletproof_vest                occurances: 2      average confidence: 0.172    confidence std: 0.047
name: swimming_trunks                 occurances: 1      average confidence: 0.074    confidence std: 0.000
name: barber_chair                    occurances: 2      average confidence: 0.499    confidence std: 0.073
name: ear                             occurances: 1      average confidence: 0.767    confidence std: 0.000
name: stinkhorn                       occurances: 2      average confidence: 0.378    confidence std: 0.060
name: breastplate                     occurances: 1      average confidence: 0.078    confidence std: 0.000
name: colobus                         occurances: 1      average confidence: 0.097    confidence std: 0.000
name: skunk                           occurances: 1      average confidence: 0.395    confidence std: 0.000
name: screen                          occurances: 1      average confidence: 0.304    confidence std: 0.000
name: monitor                         occurances: 2      average confidence: 0.185    confidence std: 0.031
name: corn                            occurances: 2      average confidence: 0.124    confidence std: 0.073
name: iron                            occurances: 1      average confidence: 0.062    confidence std: 0.000
name: seat_belt                       occurances: 1      average confidence: 0.135    confidence std: 0.000
name: miniskirt                       occurances: 1      average confidence: 0.066    confidence std: 0.000
name: Windsor_tie                     occurances: 1      average confidence: 0.053    confidence std: 0.000
name: milk_can                        occurances: 1      average confidence: 0.096    confidence std: 0.000
name: worm_fence                      occurances: 1      average confidence: 0.072    confidence std: 0.000
name: shovel                          occurances: 1      average confidence: 0.144    confidence std: 0.000
name: titi                            occurances: 1      average confidence: 0.083    confidence std: 0.000
name: speedboat                       occurances: 1      average confidence: 0.059    confidence std: 0.000
name: oscilloscope                    occurances: 1      average confidence: 0.289    confidence std: 0.000
name: bolete                          occurances: 1      average confidence: 0.114    confidence std: 0.000
name: rhinoceros_beetle               occurances: 1      average confidence: 0.152    confidence std: 0.000
name: sliding_door                    occurances: 1      average confidence: 0.157    confidence std: 0.000

 

guesses for ground

name: mantis                          occurances: 1      average confidence: 0.144    confidence std: 0.000
name: ruffed_grouse                   occurances: 1      average confidence: 0.032    confidence std: 0.000
name: vine_snake                      occurances: 3      average confidence: 0.148    confidence std: 0.135
name: cardoon                         occurances: 1      average confidence: 0.130    confidence std: 0.000
name: nematode                        occurances: 5      average confidence: 0.099    confidence std: 0.054
name: gar                             occurances: 1      average confidence: 0.111    confidence std: 0.000
name: killer_whale                    occurances: 1      average confidence: 0.242    confidence std: 0.000
name: whiptail                        occurances: 2      average confidence: 0.095    confidence std: 0.016
name: beaver                          occurances: 3      average confidence: 0.128    confidence std: 0.043
name: artichoke                       occurances: 1      average confidence: 0.237    confidence std: 0.000
name: American_chameleon              occurances: 2      average confidence: 0.330    confidence std: 0.226
name: hen-of-the-woods                occurances: 2      average confidence: 0.239    confidence std: 0.144
name: jigsaw_puzzle                   occurances: 1      average confidence: 0.059    confidence std: 0.000
name: platypus                        occurances: 3      average confidence: 0.124    confidence std: 0.068
name: marmoset                        occurances: 1      average confidence: 0.063    confidence std: 0.000
name: tailed_frog                     occurances: 2      average confidence: 0.080    confidence std: 0.019
name: sea_snake                       occurances: 1      average confidence: 0.077    confidence std: 0.000
name: submarine                       occurances: 1      average confidence: 0.294    confidence std: 0.000
name: syringe                         occurances: 1      average confidence: 0.232    confidence std: 0.000
name: bittern                         occurances: 2      average confidence: 0.119    confidence std: 0.017
name: black_stork                     occurances: 2      average confidence: 0.079    confidence std: 0.030
name: barracouta                      occurances: 1      average confidence: 0.065    confidence std: 0.000
name: garter_snake                    occurances: 1      average confidence: 0.289    confidence std: 0.000
name: spider_monkey                   occurances: 1      average confidence: 0.173    confidence std: 0.000
name: green_lizard                    occurances: 2      average confidence: 0.161    confidence std: 0.030
name: fiddler_crab                    occurances: 1      average confidence: 0.101    confidence std: 0.000
name: walking_stick                   occurances: 2      average confidence: 0.069    confidence std: 0.020
name: wallaby                         occurances: 1      average confidence: 0.029    confidence std: 0.000
name: thunder_snake                   occurances: 1      average confidence: 0.041    confidence std: 0.000
name: electric_ray                    occurances: 4      average confidence: 0.064    confidence std: 0.031
name: skunk                           occurances: 2      average confidence: 0.044    confidence std: 0.012
name: water_snake                     occurances: 2      average confidence: 0.166    confidence std: 0.063
name: bulletproof_vest                occurances: 1      average confidence: 0.080    confidence std: 0.000
name: coral_fungus                    occurances: 1      average confidence: 0.106    confidence std: 0.000
name: hatchet                         occurances: 1      average confidence: 0.044    confidence std: 0.000
name: dugong                          occurances: 1      average confidence: 0.029    confidence std: 0.000
name: titi                            occurances: 1      average confidence: 0.050    confidence std: 0.000
name: night_snake                     occurances: 1      average confidence: 0.036    confidence std: 0.000
name: missile                         occurances: 1      average confidence: 0.098    confidence std: 0.000
name: indri                           occurances: 1      average confidence: 0.046    confidence std: 0.000
name: capuchin                        occurances: 1      average confidence: 0.060    confidence std: 0.000
name: bee_eater                       occurances: 1      average confidence: 0.115    confidence std: 0.000
name: sloth_bear                      occurances: 1      average confidence: 0.044    confidence std: 0.000

 

guesses for leaf

name: ear                             occurances: 19     average confidence: 0.395    confidence std: 0.291
name: sulphur-crested_cockatoo        occurances: 1      average confidence: 0.248    confidence std: 0.000
name: European_gallinule              occurances: 3      average confidence: 0.192    confidence std: 0.053
name: vine_snake                      occurances: 2      average confidence: 0.187    confidence std: 0.050
name: cauliflower                     occurances: 1      average confidence: 0.224    confidence std: 0.000
name: porcupine                       occurances: 1      average confidence: 0.605    confidence std: 0.000
name: earthstar                       occurances: 1      average confidence: 0.078    confidence std: 0.000
name: acorn                           occurances: 2      average confidence: 0.207    confidence std: 0.082
name: flamingo                        occurances: 1      average confidence: 0.091    confidence std: 0.000
name: quill                           occurances: 4      average confidence: 0.262    confidence std: 0.203
name: mountain_bike                   occurances: 1      average confidence: 0.113    confidence std: 0.000
name: photocopier                     occurances: 1      average confidence: 0.736    confidence std: 0.000
name: American_chameleon              occurances: 2      average confidence: 0.109    confidence std: 0.023
name: black_and_gold_garden_spider    occurances: 1      average confidence: 0.114    confidence std: 0.000
name: head_cabbage                    occurances: 3      average confidence: 0.120    confidence std: 0.037
name: swab                            occurances: 1      average confidence: 0.714    confidence std: 0.000
name: greenhouse                      occurances: 1      average confidence: 0.172    confidence std: 0.000
name: pineapple                       occurances: 2      average confidence: 0.197    confidence std: 0.092
name: titi                            occurances: 1      average confidence: 0.120    confidence std: 0.000
name: beaver                          occurances: 4      average confidence: 0.100    confidence std: 0.045
name: bittern                         occurances: 4      average confidence: 0.109    confidence std: 0.031
name: meerkat                         occurances: 1      average confidence: 0.048    confidence std: 0.000
name: cucumber                        occurances: 1      average confidence: 0.091    confidence std: 0.000
name: corn                            occurances: 5      average confidence: 0.054    confidence std: 0.024
name: pot                             occurances: 1      average confidence: 0.065    confidence std: 0.000
name: artichoke                       occurances: 1      average confidence: 0.075    confidence std: 0.000
name: buckeye                         occurances: 1      average confidence: 0.160    confidence std: 0.000
name: limpkin                         occurances: 1      average confidence: 0.071    confidence std: 0.000
name: Maltese_dog                     occurances: 1      average confidence: 0.075    confidence std: 0.000
name: printer                         occurances: 1      average confidence: 0.084    confidence std: 0.000
name: thatch                          occurances: 1      average confidence: 0.083    confidence std: 0.000
name: cardoon                         occurances: 1      average confidence: 0.095    confidence std: 0.000
name: capuchin                        occurances: 1      average confidence: 0.170    confidence std: 0.000

 

guesses for stem

name: toucan                          occurances: 2      average confidence: 0.125    confidence std: 0.053
name: artichoke                       occurances: 3      average confidence: 0.139    confidence std: 0.056
name: ear                             occurances: 15     average confidence: 0.329    confidence std: 0.208
name: lumbermill                      occurances: 7      average confidence: 0.199    confidence std: 0.189
name: stinkhorn                       occurances: 4      average confidence: 0.198    confidence std: 0.104
name: bulletproof_vest                occurances: 3      average confidence: 0.474    confidence std: 0.257
name: bittern                         occurances: 4      average confidence: 0.188    confidence std: 0.056
name: king_penguin                    occurances: 1      average confidence: 0.382    confidence std: 0.000
name: harvester                       occurances: 1      average confidence: 0.085    confidence std: 0.000
name: eggnog                          occurances: 2      average confidence: 0.716    confidence std: 0.122
name: gar                             occurances: 1      average confidence: 0.270    confidence std: 0.000
name: electric_fan                    occurances: 1      average confidence: 0.081    confidence std: 0.000
name: beaver                          occurances: 1      average confidence: 0.240    confidence std: 0.000
name: worm_fence                      occurances: 2      average confidence: 0.182    confidence std: 0.075
name: titi                            occurances: 1      average confidence: 0.047    confidence std: 0.000
name: corn                            occurances: 2      average confidence: 0.244    confidence std: 0.139
name: tiger_shark                     occurances: 1      average confidence: 0.123    confidence std: 0.000
name: agaric                          occurances: 4      average confidence: 0.294    confidence std: 0.094
name: coho                            occurances: 2      average confidence: 0.362    confidence std: 0.139
name: punching_bag                    occurances: 1      average confidence: 0.244    confidence std: 0.000
name: American_egret                  occurances: 1      average confidence: 0.095    confidence std: 0.000
name: barrel                          occurances: 2      average confidence: 0.090    confidence std: 0.045
name: forklift                        occurances: 1      average confidence: 0.126    confidence std: 0.000
name: pedestal                        occurances: 1      average confidence: 0.287    confidence std: 0.000
name: quill                           occurances: 2      average confidence: 0.065    confidence std: 0.006
name: amphibian                       occurances: 2      average confidence: 0.135    confidence std: 0.052
name: tank                            occurances: 2      average confidence: 0.286    confidence std: 0.195
name: water_snake                     occurances: 1      average confidence: 0.358    confidence std: 0.000
name: photocopier                     occurances: 1      average confidence: 0.168    confidence std: 0.000
name: African_chameleon               occurances: 1      average confidence: 0.063    confidence std: 0.000
name: platypus                        occurances: 2      average confidence: 0.091    confidence std: 0.062
name: military_uniform                occurances: 4      average confidence: 0.133    confidence std: 0.100
name: candle                          occurances: 3      average confidence: 0.142    confidence std: 0.063
name: thresher                        occurances: 1      average confidence: 0.074    confidence std: 0.000
name: beaker                          occurances: 1      average confidence: 0.213    confidence std: 0.000
name: axolotl                         occurances: 2      average confidence: 0.093    confidence std: 0.032
name: plastic_bag                     occurances: 1      average confidence: 0.104    confidence std: 0.000
name: green_mamba                     occurances: 1      average confidence: 0.040    confidence std: 0.000
name: coral_fungus                    occurances: 1      average confidence: 0.230    confidence std: 0.000
name: head_cabbage                    occurances: 1      average confidence: 0.053    confidence std: 0.000
name: assault_rifle                   occurances: 1      average confidence: 0.043    confidence std: 0.000
name: measuring_cup                   occurances: 1      average confidence: 0.085    confidence std: 0.000
name: ice_lolly                       occurances: 1      average confidence: 0.163    confidence std: 0.000
name: solar_dish                      occurances: 1      average confidence: 0.071    confidence std: 0.000
name: European_gallinule              occurances: 1      average confidence: 0.035    confidence std: 0.000
name: crash_helmet                    occurances: 1      average confidence: 0.079    confidence std: 0.000
name: pineapple                       occurances: 1      average confidence: 0.176    confidence std: 0.000
```
2)
```
Train on 178 samples, validate on 67 samples
Epoch 1/20
178/178 [==============================] - 17s 98ms/step - loss: 11.2994 - acc: 0.2584 - val_loss: 13.4718 - val_acc: 0.1642
Epoch 2/20
178/178 [==============================] - 13s 72ms/step - loss: 11.0187 - acc: 0.2921 - val_loss: 11.0662 - val_acc: 0.3134
Epoch 3/20
178/178 [==============================] - 12s 70ms/step - loss: 8.8974 - acc: 0.4101 - val_loss: 6.7081 - val_acc: 0.5075
Epoch 4/20
178/178 [==============================] - 15s 84ms/step - loss: 8.7830 - acc: 0.4157 - val_loss: 6.6183 - val_acc: 0.5672
Epoch 5/20
178/178 [==============================] - 15s 82ms/step - loss: 7.8267 - acc: 0.4888 - val_loss: 9.0165 - val_acc: 0.4328
Epoch 6/20
178/178 [==============================] - 17s 94ms/step - loss: 6.7493 - acc: 0.5618 - val_loss: 6.9330 - val_acc: 0.5373
Epoch 7/20
178/178 [==============================] - 15s 85ms/step - loss: 6.9399 - acc: 0.5506 - val_loss: 8.9599 - val_acc: 0.4328
Epoch 8/20
178/178 [==============================] - 13s 74ms/step - loss: 6.5241 - acc: 0.5562 - val_loss: 8.2859 - val_acc: 0.4627
Epoch 9/20
178/178 [==============================] - 14s 79ms/step - loss: 6.0204 - acc: 0.6067 - val_loss: 6.3230 - val_acc: 0.5522
Epoch 10/20
178/178 [==============================] - 14s 77ms/step - loss: 6.9524 - acc: 0.5393 - val_loss: 6.4682 - val_acc: 0.5821
Epoch 11/20
178/178 [==============================] - 13s 75ms/step - loss: 5.0455 - acc: 0.6573 - val_loss: 6.8423 - val_acc: 0.5672
Epoch 12/20
178/178 [==============================] - 13s 71ms/step - loss: 5.3021 - acc: 0.6404 - val_loss: 4.9032 - val_acc: 0.6866
Epoch 13/20
178/178 [==============================] - 13s 74ms/step - loss: 5.6250 - acc: 0.6236 - val_loss: 6.2336 - val_acc: 0.5970
Epoch 14/20
178/178 [==============================] - 15s 82ms/step - loss: 5.9581 - acc: 0.6011 - val_loss: 6.7411 - val_acc: 0.5821
Epoch 15/20
178/178 [==============================] - 12s 66ms/step - loss: 4.7660 - acc: 0.6573 - val_loss: 5.7175 - val_acc: 0.6119
Epoch 16/20
178/178 [==============================] - 13s 75ms/step - loss: 4.3269 - acc: 0.7022 - val_loss: 5.0397 - val_acc: 0.6716
Epoch 17/20
178/178 [==============================] - 13s 71ms/step - loss: 4.0705 - acc: 0.7360 - val_loss: 5.0519 - val_acc: 0.6866
Epoch 18/20
178/178 [==============================] - 13s 74ms/step - loss: 4.7396 - acc: 0.6798 - val_loss: 5.5653 - val_acc: 0.6418
Epoch 19/20
178/178 [==============================] - 12s 66ms/step - loss: 3.7560 - acc: 0.7528 - val_loss: 4.1656 - val_acc: 0.7313
Epoch 20/20
178/178 [==============================] - 12s 66ms/step - loss: 4.0352 - acc: 0.7247 - val_loss: 4.4792 - val_acc: 0.7015
No of errors = 20/67
```
