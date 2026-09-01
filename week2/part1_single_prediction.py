blade_angle = [8.5, 9.5, 9.9, 9.0]
weight = 0.5

def neural_network(input, input_weight):
    return input * input_weight

for angle in blade_angle:
    prediction = neural_network(angle, weight)
    print(f"Prediction for {angle}: {prediction}")

'''
1. What does the weight do?: scales each blade angle down by a half, essentially making it 
less "important" for the predicition.
2. What would happen if you changed it from 0.5 to 2.0?: It would increase the angles
influence in the neural network, making it more sensitive to the blade's angle changes.
3. What if you set it to a negative number?: It would inverse the prediction meaning the higher 
the blade angle the lower the prediction's value.


'''