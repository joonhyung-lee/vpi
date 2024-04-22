from collections import defaultdict

def get_stacked_order(object_positions, object_names, threshold=0.15):
    # Group objects by similar XY coordinates
    xy_groups = defaultdict(list)
    for pos, name in zip(object_positions, object_names):
        x, y, _ = pos
        key = tuple(round(coord / threshold) for coord in (x, y))  # Grouping key
        xy_groups[key].append((pos, name))

    # Determine stacking order within each XY group
    stacked_orders = []
    for group in xy_groups.values():
        group.sort(key=lambda x: x[0][2])  # Sort by Z-coordinate
        stacked_orders.append([name for _, name in group])

    return stacked_orders

def extract_info(object_name):
    parts = object_name.split('_')
    shape = parts[1]
    color = parts[2]
    return shape, color

def is_stacked(coord1, coord2, threshold=0.05):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2

    return abs(x1 - x2) < threshold and abs(y1 - y2) < threshold and z1 != z2

def preference_criteria(color_match_count, shape_match_count, stack_count, stacked_order):
    # Define weights
    weight_color = 1
    weight_shape = 1
    weight_stack = 0.5

    # Color preference
    color_score = sum(color_match_count.values()) * weight_color
    # Shape preference
    shape_score = sum(shape_match_count.values()) * weight_shape
    # Stack preference
    stack_score = stack_count * weight_stack
    # Calculate total preference score
    total_score = color_score + shape_score + stack_score

    if color_score > 3:
        color_preference = "High preference for color"
    elif color_score == 3:
        color_preference = "Slightly high preference for color"
    elif color_score < 3:
        color_preference = "Low preference for color"
    
    if shape_score > 3:
        shape_preference = "High preference for shape"
    elif shape_score == 3:
        shape_preference = "Slightly high preference for shape"
    elif shape_score < 3:
        shape_preference = "Low preference for shape"

    if stack_score > 2:
        stack_preference = "High preference for stacking"
    elif stack_score == 2:
        stack_preference = "Slightly high preference for stacking"
    elif stack_score < 2:
        stack_preference = "Low preference for stacking"
    
    preference = "User Preference: " + color_preference + ", " + shape_preference + ", " + stack_preference

    return total_score, preference

def transform_relation(relation):
    transformations = {
        "left_of": "left",
        "right_of": "right",
        "front of": "front",
        "behind of": "behind",
        "stacked on": "stacked",
        # Add more transformations as needed
    }
    return transformations.get(relation, relation)

def calculate_interaction_accuracy(response_json, interaction_info, category):
    if response_json is None or "images" not in response_json:
        print("Warning: response_json is None or missing 'images' key.")
        return 0
    if interaction_info is None or "interaction" not in interaction_info:
        print("Warning: interaction_info is None or missing 'interaction' key.")
        return 0

    # total_interactions = len(response_json["images"])
    total_interactions = min(len(response_json["images"]), len(interaction_info["interaction"]))
    correct_matches = 0

    if category == 'color':
        for i in range(total_interactions):
            response = response_json["images"][f"interaction_{i+1}"]
            ground_truth = interaction_info["interaction"][i][0].split(',')
            print(f"Response: {response}")
            print(f"Ground Truth: {ground_truth}")
            
            involved_object = ground_truth[0].strip().lower()
            ground_truth_target = ground_truth[1].strip().lower()
            ground_truth_relation = ground_truth[2].strip().lower()

            # Check if the involved objects match
            if involved_object in response['involved_objects']:
                response_relations = response['spatial_relations'].get(involved_object, [])

                # Transforming 'stacked' to 'inside' for comparison
                transformed_ground_truth_relation = "inside" if ground_truth_relation == "stacked" else ground_truth_relation
                transformed_ground_truth_relation = f"{transformed_ground_truth_relation}({ground_truth_target})".lower()

                # Check if spatial relations match
                if transformed_ground_truth_relation in response_relations:
                    correct_matches += 1

        accuracy = correct_matches / total_interactions
    elif category == 'shape':
        for i in range(total_interactions):
            response = response_json["images"][f"interaction_{i+1}"]
            ground_truth = interaction_info["interaction"][i][0].split(',')
            print(f"Response: {response}")
            print(f"Ground Truth: {ground_truth}")
            
            involved_object = ground_truth[0].strip().lower()
            ground_truth_target = ground_truth[1].strip().lower()
            ground_truth_relation = ground_truth[2].strip().lower()
            # print(f"ground_truth_target: {ground_truth_target}")
            # print(f"ground_truth_relation: {ground_truth_relation}")
            # print(f"response['involved_objects']: {response['involved_objects']}")

            # Check if the involved objects match
            if involved_object in response['involved_objects']:
                response_relations = response['spatial_relations'].get(involved_object, [])

                # Transform ground truth relation
                transformed_ground_truth_relation = transform_relation(ground_truth_relation)
                transformed_ground_truth_relation = f"{transformed_ground_truth_relation}({ground_truth_target})".lower()
                print(f"transformed_ground_truth_relation: {transformed_ground_truth_relation}")
                print(f"response_relations: {response_relations}")
                # Check if spatial relations match
                if transformed_ground_truth_relation in response_relations:
                    correct_matches += 1

        accuracy = correct_matches / total_interactions
    elif category == 'stack':
        for i in range(total_interactions):
            response = response_json["images"][f"interaction_{i+1}"]
            ground_truth = interaction_info["interaction"][i][0].split(',')
            print(f"Response: {response}")
            print(f"Ground Truth: {ground_truth}")

            involved_object = ground_truth[0].strip().lower()
            ground_truth_target = ground_truth[1].strip().lower()
            ground_truth_relation = ground_truth[2].strip().lower()

            # Check if the involved objects match
            if involved_object in response['involved_objects']:
                response_relations = response['spatial_relations'].get(involved_object, [])
                extracted_relations = [relation.split('(')[0] for relation in response_relations]
                print(f"Extracted Relations: {extracted_relations}")

                # Transform 'above' to 'stacked_on' for matching
                if ground_truth_relation == "above":
                    ground_truth_relation = "stacked_on"

                # Construct the expected relation string for comparison
                expected_relation = f"{ground_truth_relation}({ground_truth_target})".lower()

                # Check if any of the spatial relations match
                if any(expected_relation in relation for relation in response_relations):
                    correct_matches += 1

        accuracy = correct_matches / total_interactions
    return accuracy

#     elif category == 'stack':
#         for i in range(total_interactions):
#             response = response_json["images"][f"interaction_{i+1}"]
#             ground_truth = interaction_info["interaction"][i][0].split(',')
#             print(f"Response: {response}")
#             print(f"Ground Truth: {ground_truth}")
            
#             involved_object = ground_truth[0].strip().lower()
#             ground_truth_target = ground_truth[1].strip().lower()
#             ground_truth_relation = ground_truth[2].strip().lower()

#             # Check if the involved objects match
#             if involved_object in response['involved_objects']:
#                 response_relations = response['spatial_relations'].get(involved_object, [])
#                 extracted_relations = [relation.split('(')[0] for relation in response_relations]
#                 # Transforming 'stacked' to 'inside' for comparison
#                 transformed_ground_truth_relation = extracted_relations if ground_truth_relation == "above" else ground_truth_relation
#                 print(f"Extracted Relations: {extracted_relations}")
#                 transformed_ground_truth_relation = f"{transformed_ground_truth_relation}({ground_truth_target})".lower()
#                 print(f"transformed_ground_truth_relation: {transformed_ground_truth_relation}")

#                 # Check if spatial relations match
#                 if transformed_ground_truth_relation in response_relations:
#                     correct_matches += 1

#         accuracy = correct_matches / total_interactions
#     return accuracy
