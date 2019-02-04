# SpeedCar-Prediction

Predicting velocity of a car from the front dashcam video.

Still need to implement the images prep, augmenting them with resize, change of brightness ecc..
The NN looks nice tho.

TODOs:
* implementing the normalization, when I switched to RGB it broke
* Generation of more DATAs (should use the fit from generator but it seems a lot of work, last thing to do)

Changed some params, increased the dropout percent by 0.05 and the sequence lenght (16 -> 20).
Now after 15 epochs it gets aounrd 6 training loss and around 7-8 validation loss. It doesn't seem to overfit yet.
Maybe normalization will help for the last push to get a better loss.
