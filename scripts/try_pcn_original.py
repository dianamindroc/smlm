from dataset.SimulatorDataset import DNAOrigamiSimulator

# Dye properties dictionary
dye_properties_dict = {
    'Alexa_Fluor_647': {'density_range': (10, 50), 'blinking_times_range': (10, 50), 'intensity_range': (500, 5000), 'precision_range' : (0.5, 2.0)},
    # ... (other dyes)
}

# Choose a specific dye
selected_dye = 'Alexa_Fluor_647'
selected_dye_properties = dye_properties_dict[selected_dye]
# Define structure parameters and noise conditions

# Initialize the DNAOrigamiSimulator
box_dataset = DNAOrigamiSimulator(num_samples=200, structure_type= 'box', dye_properties=selected_dye_properties, augment=True, remove_corners=True)

# Verify the simulator is correctly initialized
#print(len(simulator), simulator[0])  # Display the length of the simulator and the first item

boxes = []
for box in box_dataset:
    boxes.append(box[1])