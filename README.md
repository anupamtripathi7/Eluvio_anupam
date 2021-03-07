# Eluvio_anupam
Eluvio coding challenge

Abstract:<br>
The given task was to perform scene segmentation on the MovieScene dataset. 
A scene segment is made of one or more clips and therefore by predicting which all clips are scene boundries, the given task can be broken down to binary classification task for n-1 clips if n is the total number of clips in the movie. 
This task was done with a Mean Average Precision (mAP) of 0.723 and a Mean Maximum IoU (mean Miou) of 0.7263. The resultant f1 score of the classification task was 0.723.

Methodology:<br>
The four extracted features of place, action, cast and audio give information about all the given clips of a movie. 
A significant change in all or majority of these four could mean a scene change. For any of these 4 features, consider a feature matrix of c x f, with c clips and f dimensions. 
To compare two consecutive clips, we can simply take the cosine similarity between those two clips. 
For the resultant feature vector, the features for clip i will be given by
<br>
c'i = abs(ci - ci+1)
<br>
Where ci is the ith row of the original feature matrix, thus converting a c x f matrix to c x 1. 
Now this is by considering just one nearby clip. 
By considering k neighboring clips, we get a resultant feature matrix of c x k by having padding wherever necessary. 
c'i = [abs(ci - ci-k/2), abs(ci - ci-k/2+1), ... abs(ci - ci+k/2-1), abs(ci - ci+k/2)]
<br>
With 4 such matrices, we get a resultant feature matrix of c x 4*k, which is passed through an LSTM to get the resultant predictions

Results:<br>
The results using the evaluation code is shown bellow:
<br><br>
Scores: {
    "AP": 0.7142567241228993,
    "mAP": 0.7230931137094851,
    "Miou": 0.7263464799002725,
    "Precision": 0.7521324348635972,
    "Recall": 0.7089855947997421,
    "F1": 0.7230989675955999
}
<br><br>
The split used was 80-20 for training but the above given results are on the entire dataset.

Conclusion:<br>
Transforming the problem into a binary classification task resulted in a highly imbalanced dataset but the model was able to generalize it to a good extent. The code and the trained model are available in the github repository. 
