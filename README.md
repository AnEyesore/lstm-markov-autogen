# Automatic Music Generation using Machine Learning on MAESTRO v3.0.0

[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro#v300) is a dataset composed of upwards of 1200 classical piano performances, both in wav and in MIDI formats. The purpose of this project was to see if a machine learning model could be trained in a way as to create "compositions" that mimic the styles of specific composers. To determine this objectively, a strong classifier to categorize the pieces of the top 4 most prolific composers in the Maestro dataset. Then, 8 generation models were trained (2 methods, LSTM and Markov Chain, per composer). Each model generated 100 pieces, and was run through the classifier to determine how close of a match these models were to the actual composers.

### Feature Engineering

Feature Engineering was performed using the [music21](https://web.mit.edu/music21/) library developed at MIT. Methods in this library can, among other functions, analyse MIDI files and extract relevant numerical information from them. The command
```python
f = features.allFeaturesAsList()
print(f)
```
returns 92 features, which was too many features to train a classifier on, so several methods of feature seleciton were used.

1) Pre-selection -- Several of the features extracted through music21 referred to different instrumental voices. Since the dataset was limited to piano performances only, these could safely be ignored. This removed 6 features.
2) Boruta -- The Boruta algorithm was used to determine which features had a minimal impact on predictive power and could safely be ignored. This removed 26 features.
3) Covariance Analysis --  Highly covariant (cov > 0.5) features were identified, and one of each highly-covariant pair was removed (randomly). This removed 21 features.

The final feature count following engineering was 39, as seen below.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/35a27bd5-236f-4351-9d11-ee5355ad227a)

A quick PCA decomposition showed that the data appeared to be (relatively) cleanly separable, and that the feautres that remained were likely adequate to create a well-functioning classifier.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/32106b9c-9fa8-48f3-ad03-887e82d0229e)

### Classification

A classifier was built using an AutoML library called [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html). The data was separated randomly into a training set (80%) and a holdout test set (20%). In each case, the classes were well balanced.

The final model was a Stacked Ensemble classifier which displayed an error rate of 3% on the training data and 1.4% on the holdout sample. Exact confusion matrices can be seen below.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/3e8447bd-74c5-41ac-b369-30fd200b47bf)

This is a very strong classifier given the extracted features. 

### Model 1 - LSTM Neural Network

The LSTM used Tensorflow with Keras API. Each model was trained on sequences of 50 notes with a maximum of 50 epochs, though most models stopped around 20 training epochs to prevent overfitting. The LSTM had three outputs: pitch, step and duration. Since losses in initial trials were overwhelmingly driven by pitch, they were scaled down to properly modify step and duration as well. The model was constructed as below:

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/d16f10b9-d569-462c-8272-5c88ee0822f9)

After training, 100 randomly selected seed sequences of 50 notes were taken and used to predict short compositions. This was done for each composer in the list, resulting in 400 compositions total. These compositions were then feature engineered as the base data, and run through the classifier.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/45c348ac-ee35-4523-bf11-941a782c6b01)

These results are not encouraging. 29.5% accuracy on the generated set is only marginally better than the expected outcome from random guessing. Interestingly, however, the PCA decomposition shows that the LSTM models did create stylistically different compositions. All of the generated MIDIs for these models can be found in their respective folders (chopAIn -> Chopin, JSBot -> J.S. Bach, schuBot -> Schubert, lstm_von_Bothooven -> Beethoven).

### Model 2 - Markov Chain

For the Markov Chain models, transition probabilities were calculated from one note to the next for each composer.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/9a4ab4b8-0788-45ed-850b-7531771340ff)

To generate a MIDI, a seed note was selected and the following note was selected based on these transition probabilities. This method is probabilistic and not tuned for architecture in the slightest, but rather aims to fool a classifier by using actual existing intervals and transitions. The result is that all of the generated MIDIs sound like random jumbles of notes. Interestingly, however, the classifier proved better at identifying these, with a 38.5% accuracy rate.

![image](https://github.com/AnEyesore/lstm-markov-autogen/assets/160987733/9071346b-7d0e-4a13-a3a6-9be78e742c5f)

Once again, the MIDIs can be found in their respective folders (markov_Chop -> Chopin, markov_Bach -> J.S. Bach, markov_Schu -> Schubert, markov_Beet -> Beethoven).

### Conclusions and Consideration for Future Study

The Markov Chain-generated compositions proved more effective at fooling the classifier than the LSTM pieces did. This is likely due to the nature of the feature engineering caring more substantially about the specific intervals between notes rather than anything stylistic. Notably absent from this study as well is the existence of chords-- where the models predicted only a single note at a time, the addition of chord structures could more accurately mimic the styles of the actual composers. That said, that is a very complex topic that has had studies done on its own.

A second topic of consideration would be the separation of the two moving lines. Piano performances use two hands and ostensibly can play up to 10 notes at a time. It may be interesting to separate out left hand vs. right hand, and training two sets of models per composer to model the activities and movements of each hand.

Finally, it would be interesting to consider a more architectural approach to creating a composition-- devising a set of rules that each composition should follow to sound pleasign to a listener. As is, while these models may be mathematically correct to an extent, one cannot deny that they sound like uninspired D-list horror movie soundtracks. In the end, perhaps it is the combination of the three considerations above that could lead to true automatic composition in the style of these composers.
