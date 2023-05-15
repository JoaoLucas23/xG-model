# xG-model

## Description

This is a model to predict the expected goals of a shot in a soccer game. This model uses Logistic Regression to predict the expected goals of a shot. The model uses Wyscout open data to train and test the model. In this model, only freekick shots and regular shots were considered to train. To consider penalties, the percentage of penalties taken that resulted in a goal was calculated and set as the xg value of a penalty shot. This value varies from league to league, being 0.7168 in the Spanish League and 0.7 in the English League.

## Model

### Logistic Regression

The Logistic Regression model is a classification model that uses the logistic function to predict the probability of a binary outcome. In this case, the binary outcome is whether the shot resulted in a goal or not. This was the best choice for this model because it is a simple model and the results are easy to interpret. The model had an accuracy of 89.92% in the train data and 89.58% in the test data.

### Features

- Distance from goal
- Angle from goal
- Angle squared
- Distance times angle
- Body part
- Shot type
- Previous event
- Shot result

## Data

### Train Data

To train the model, all shots and freekick shots in the Italian, French and German League on the 17/18 season were used. This train data resulted in 25266 total shots.

### Test Data

To test the model, all shots and freekick shots in the English and Spanish League on the 17/18 season were used. This test data resulted in 17233 total shots. Additionally, penalty shots with the fixed xg value mentioned before were added.

## Files

- loader.py: loads the event data from the local directory and transforms to the SPADL format
- features.py: calculates and gets the features of the shots
- training.py: trains the model
- predictions.py: predicts the xg of the shots
- main.py: main file to run the model
- wyDataLoader.py: loads the additional data such as Players and Teams info and minutes played per player per game


## Results
### Train Data
|feature|coeficientes|
|---|---|
|angle|0.03|
|distance|-0.19|
|distance2|0.00|
|angle2|-0.00|
|dist_angle|0.00|
|prev_action_clearance|-0.29|
|prev_action_corner_crossed|-0.43|
|prev_action_corner_short|-0.09|
|prev_action_cross|-0.17|
|prev_action_dribble|-0.02|
|prev_action_foul|0.34|
|prev_action_freekick_crossed|-0.17|
|prev_action_freekick_short|0.02|
|prev_action_goalkick|-0.01|
|prev_action_interception|-0.10|
|prev_action_keeper_save|0.33|
|prev_action_pass|0.19|
|prev_action_shot|-0.01|
|prev_action_tackle|0.02|
|prev_action_take_on|0.17|
|prev_action_throw_in|-0.03|
|bodypart_name_foot|0.32|
|bodypart_name_head/other|-0.57|
|type_name_shot|-0.59|
|type_name_shot_freekick|0.34|


### Test Data
|player_name|total shots|xg|goal|minutes_played|xg_per_90|xg_per_shot|goal_per_90|goal_per_shot|
|---|---|---|---|---|---|---|---|---|
|Cristiano Ronaldo|170|26.33|26|2355|1.01|0.99|0.15|0.15|
|H. Kane|175|25.52|29|3201|0.72|0.82|0.15|0.17|
|L. Messi|193|25.16|34|3108|0.73|0.98|0.18|0.18|
|Mohamed Salah|142|19.69|32|2995|0.59|0.96|0.23|0.23|
|Gerard Moreno|100|15.95|16|3531|0.41|0.41|0.16|0.16|
|R. Sterling|80|15.54|18|2697|0.52|0.60|0.22|0.23|
|Iago Aspas|94|15.37|22|3038|0.46|0.65|0.23|0.23|
|R. Lukaku|80|15.11|16|2998|0.45|0.48|0.20|0.20|
|Stuani|69|14.82|21|2785|0.48|0.68|0.30|0.30|
|J. Vardy|66|14.02|20|3370|0.37|0.53|0.30|0.30|

## To Do
* Add feature Assisted
