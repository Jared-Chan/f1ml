# Lap-by-Lap Prediction Model

Lap-by-lap predictions are extremely useful for determining to best strategies before and during a race. While other Formula One prediction models exist, most of them focus on predicting the winner of a race, as opposed to lap times and positions of all drivers. This model tries to do that using readily available data and training methods.

## Data Preparation

The model was trained using data from the [Ergast Developer API](http://ergast.com/mrd/). While the data is comprehensive, it is not in a convenient format for our model. After looking through the data, I organized them into Pandas Data Frames so that they can easily be stored and loaded from CSV files. Each row contained information about twenty drivers, representing one lap of data, and one file contained data of one race.

Information like lap times, driver IDs, and positions were stored as numbers. This format was easy to read for humans and can be converted into PyTorch tensors without much hassle. However, the models failed to train effectively and they tended to over-fit. One reason might be that some data such as driver IDs range from 1 to over 800, while other data such as positions range from 0 to 20. Even after scaling our data, our models had a hard time learning. Using numbers to represent drivers, the models could not differentiate drivers well.

So, instead of using integers, I converted each number into a one-hot representation. I chose to use data from 2001 onwards to not only include all races that involved current drivers but also cut down on the number of drivers needed to be represented. Instead of 770 drivers, I only needed to encode about 120. To allow easier future addition of drivers, I chose to use a vector of length 130 for driver embedding.

After making the encoding for driver IDs, I also made encoding for lap times. Instead of a floating-point number, lap time is represented using 32 bits. They map to values from 0.0 to 399.9 with a precision of one decimal place. Using this representation seemed very wasteful but after some experimentations, it yields more accurate results. This may be because each digit has equal importance, instead of the first one or two digits being dominant, so after applying the loss function and back-propagating, all digits are equally optimized. This may also be because since there are now 32 bits, there is just simply more information to be learned, so naturally, the model performs better. I chose 32 bits as a compromise between higher precision and lower input size.

Pit stop prediction turned out to be the trickiest part. At first, I let the model predict whether each driver is pitting in the next lap. However, out of 60 laps, a driver may only pit once or twice, making our samples very unbalanced. The model can predict drivers to be not pitting every lap and the loss function would still report a low loss. It is also not sensible to upsample our 'pit stop laps' since they would not represent real situations. Upsampling would also go against predicting pit stop strategies. I could choose not to include pit stop prediction, but since many position changes occur during pit stops, I hoped that the model picks up on these changes and make predictions accordingly. Finally, I decided to let the model predict the number of laps until a driver makes a pit stop. Now, Data are abundant. Also, since the length of each circuit can be very different, instead of predicting the number of laps, the model outputs a fraction that represents the number of laps/ total number of laps.

Various incidents, like collisions and mechanical errors, also have a huge impact on lap times and retirements. These incidents are recorded in detail in the Ergast database. However, there are over 130 types of these situations so it would make inputs even larger to dedicate 130 bits for statuses. Furthermore predicting statuses is not the focus of this model. Hence, I analyzed all the incidents from 2001 to 2020 and grouped them into six categories. Then I only need 6 bits to encode status. Another problem, like with pit stops, was that the samples are unbalanced. But unlike pit stops, accidents are not bound to happen. We cannot simply guess how many laps until the next accident. So, I added an extra position, position 21, which represents whether a driver has retired from the race. When a driver is predicted to be at position 21, we can then interpret the statuses differently. That is, ignoring the bit that indicates 'no problem.'

Information like driver standings and constructor standings can indicate how good a driver or constructor is in the current season. I chose to include these two pieces of information but used a simpler representation since they are not crucial. I used 3 bits to represent driver standing and 2 bits to represent constructor standing for each driver. They tell the model the general position in the drivers' and constructors' championship instead of the exact place.

Finally, I added an extra bit that is always zero when training. This is the 'randomness' bit that allows us to add some unpredictability to our model when using it for predictions.

I also wanted to include other kinds of information, such as weather status, tire types, and safety car laps, but I could not find much information and I wanted to try training a model with what I had.

At the end this is what the input and output of the model look like: (with their respective sizes)

Input (4051):
- circuit ID (130)
- Lap number / total number of laps (1)
- For each driver:
- - Driver standing (3)
- - Constructor standing (2)
- - Position (21)
- - Number of laps until pit stop / total number of laps (1)
- - Status (6)
- - Lap time (32)
- - Random (1)

Output (1200):
- For each driver:
- - Position (21)
- - Number of laps until pit stop / total number of laps (1)
- - Status (6)
- - Lap time (32)


## Model

The model is a long short-term memory network (LSTM), which is ideal for processing sequences of data. I trained with PyTorch using Google Colaboratory. After much experimentation, I found that a simpler model is more accurate for the dataset I have. Increasing the number of layers, or the size of the hidden layers does not always improve accuracy but usually leads to over-fitting. Finally, I settled for a hidden size of 1600 and a layer number of 2 for the LSTM. These parameters give a model that predicts sensibly, maybe almost as well as a human being. Using a smaller size also makes the model small enough to be put on a website.

Since I only have information on pit stops from 2012 onwards, I trained two models, one with all the data starting from 2001 and one with data starting from 2012. The first one is more accurate overall while the second one is more accurate in predicting pit stops.

Since the sequence of laps is important and each race should be one whole unit, feeding data to the model is not as simple as shuffling and creating batches. I made a dataset class that allows intuitive control of getting data from a specified year, race, and lap number to facilitate model training.

## Code for all of the above

- `RacePrediction.ipynb`: Data preparation and the first version of the model
- `RacePrediction2.ipynb`: Making of all the encoding, the second version of the model, new format of data
- `RacePrediction3.ipynb`: New pit stop prediction method
- `RacePrediction5.ipynb`: Final versions of everything and a much more efficient dataset class.
- `splitfiles.ipynb`: Code for splitting the models' state dictionaries into smaller chunks so that they can be uploaded to GitHub.
- `formatting.ipynb`: Code for getting the latest qualifying lap times from the Ergast API.

To view the notebooks online:
- [RacePrediction](https://colab.research.google.com/drive/1hl3SCef_1z_JthlxZQ9kH5coZwkFoVsd?usp=sharing)
- [RacePrediction2](https://colab.research.google.com/drive/1ZGqolGigHaxgB-iYLTJwgOE3DZ5RN9xy?usp=sharing)
- [RacePrediction3](https://colab.research.google.com/drive/16xzjllKOPqA1TPBvvV7goyfaQ6OVsejI?usp=sharing)
- [RacePrediction5](https://colab.research.google.com/drive/1CkJh1JWBi9KB-9PkKNEA9XSy-jgHfo63?usp=sharing)
- [formatting](https://colab.research.google.com/drive/16ASupTWkqUASTbmqwceK4tTdl0rMLDea?usp=sharing)
