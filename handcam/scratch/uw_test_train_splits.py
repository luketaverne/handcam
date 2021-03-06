import pickle

test_splits = [
    [
        'apple_3', 'ball_1', 'banana_4', 'bell_pepper_6', 'binder_3', 'bowl_5', 'calculator_1', 'camera_2', 'cap_1',
        'cell_phone_2', 'cereal_box_3', 'coffee_mug_2', 'comb_1', 'dry_battery_3', 'flashlight_3', 'food_bag_1',
        'food_box_5', 'food_can_11', 'food_cup_2', 'food_jar_4', 'garlic_2', 'glue_stick_5', 'greens_2', 'hand_towel_2',
        'instant_noodles_3', 'keyboard_1', 'kleenex_4', 'lemon_4', 'lightbulb_3', 'lime_3', 'marker_1', 'mushroom_2',
        'notebook_3', 'onion_4', 'orange_4', 'peach_3', 'pear_8', 'pitcher_3', 'plate_4', 'pliers_3', 'potato_1',
        'rubber_eraser_3', 'scissors_4', 'shampoo_3', 'soda_can_2', 'sponge_5', 'stapler_6', 'tomato_8', 'toothbrush_3',
        'toothpaste_5', 'water_bottle_2'
    ],
    [
        'apple_3', 'ball_1', 'banana_4', 'bell_pepper_3', 'binder_3', 'bowl_6', 'calculator_2', 'camera_2', 'cap_4',
        'cell_phone_2', 'cereal_box_3', 'coffee_mug_8', 'comb_5', 'dry_battery_4', 'flashlight_5', 'food_bag_2',
        'food_box_9', 'food_can_7', 'food_cup_3', 'food_jar_4', 'garlic_2', 'glue_stick_1', 'greens_1', 'hand_towel_2',
        'instant_noodles_5', 'keyboard_1', 'kleenex_4', 'lemon_3', 'lightbulb_2', 'lime_4', 'marker_4', 'mushroom_1',
        'notebook_1', 'onion_2', 'orange_1', 'peach_2', 'pear_1', 'pitcher_2', 'plate_6', 'pliers_2', 'potato_5',
        'rubber_eraser_4', 'scissors_4', 'shampoo_4', 'soda_can_3', 'sponge_11', 'stapler_5', 'tomato_5',
        'toothbrush_3', 'toothpaste_4', 'water_bottle_9'
    ],
    [
        'apple_2', 'ball_4', 'banana_4', 'bell_pepper_1', 'binder_3', 'bowl_1', 'calculator_4', 'camera_2',
        'cap_4', 'cell_phone_3', 'cereal_box_5', 'coffee_mug_8', 'comb_5', 'dry_battery_5', 'flashlight_1',
        'food_bag_2', 'food_box_5', 'food_can_11', 'food_cup_5', 'food_jar_5', 'garlic_2', 'glue_stick_4',
        'greens_2', 'hand_towel_4', 'instant_noodles_3', 'keyboard_5', 'kleenex_2', 'lemon_2', 'lightbulb_3',
        'lime_1', 'marker_9', 'mushroom_2', 'notebook_2', 'onion_4', 'orange_4', 'peach_2', 'pear_2',
        'pitcher_1', 'plate_2', 'pliers_3', 'potato_5', 'rubber_eraser_1', 'scissors_1', 'shampoo_2',
        'soda_can_4', 'sponge_4', 'stapler_7', 'tomato_8', 'toothbrush_1', 'toothpaste_1', 'water_bottle_2'
    ],
    [
        'apple_5', 'ball_6', 'banana_1', 'bell_pepper_2', 'binder_1', 'bowl_6', 'calculator_1', 'camera_3', 'cap_3',
        'cell_phone_5', 'cereal_box_3', 'coffee_mug_7', 'comb_2', 'dry_battery_6', 'flashlight_4', 'food_bag_7',
        'food_box_4', 'food_can_8', 'food_cup_1', 'food_jar_2', 'garlic_5', 'glue_stick_4', 'greens_3', 'hand_towel_4',
        'instant_noodles_4', 'keyboard_3', 'kleenex_5', 'lemon_3', 'lightbulb_1', 'lime_4', 'marker_2', 'mushroom_3',
        'notebook_1', 'onion_6', 'orange_1', 'peach_1', 'pear_2', 'pitcher_1', 'plate_7', 'pliers_6', 'potato_1',
        'rubber_eraser_1', 'scissors_3', 'shampoo_6', 'soda_can_6', 'sponge_10', 'stapler_1', 'tomato_5',
        'toothbrush_1', 'toothpaste_1', 'water_bottle_3'
    ],
    [
        'apple_4', 'ball_2', 'banana_4', 'bell_pepper_1', 'binder_2', 'bowl_5', 'calculator_2', 'camera_1', 'cap_3',
        'cell_phone_5', 'cereal_box_3', 'coffee_mug_6', 'comb_3', 'dry_battery_6', 'flashlight_5', 'food_bag_6',
        'food_box_1', 'food_can_12', 'food_cup_3', 'food_jar_4', 'garlic_1', 'glue_stick_1', 'greens_2', 'hand_towel_4',
        'instant_noodles_4', 'keyboard_3', 'kleenex_3', 'lemon_6', 'lightbulb_1', 'lime_2', 'marker_1', 'mushroom_3',
        'notebook_2', 'onion_5', 'orange_3', 'peach_3', 'pear_2', 'pitcher_1', 'plate_1', 'pliers_2', 'potato_4',
        'rubber_eraser_2', 'scissors_1', 'shampoo_4', 'soda_can_4', 'sponge_7', 'stapler_5', 'tomato_1', 'toothbrush_2',
        'toothpaste_3', 'water_bottle_10'
    ],
    [
        'apple_3', 'ball_3', 'banana_3', 'bell_pepper_1', 'binder_1', 'bowl_6', 'calculator_3', 'camera_3', 'cap_4',
        'cell_phone_1', 'cereal_box_3', 'coffee_mug_7', 'comb_2', 'dry_battery_4', 'flashlight_3', 'food_bag_6',
        'food_box_8', 'food_can_4', 'food_cup_1', 'food_jar_1', 'garlic_4', 'glue_stick_2', 'greens_2', 'hand_towel_5',
        'instant_noodles_6', 'keyboard_1', 'kleenex_3', 'lemon_3', 'lightbulb_3', 'lime_4', 'marker_3', 'mushroom_1',
        'notebook_2', 'onion_3', 'orange_2', 'peach_2', 'pear_3', 'pitcher_2', 'plate_1', 'pliers_2', 'potato_6',
        'rubber_eraser_3', 'scissors_3', 'shampoo_2', 'soda_can_6', 'sponge_3', 'stapler_8', 'tomato_1', 'toothbrush_2',
        'toothpaste_3', 'water_bottle_1'
    ],
    [
        'apple_2', 'ball_2', 'banana_3', 'bell_pepper_3', 'binder_1', 'bowl_4', 'calculator_5', 'camera_1', 'cap_2',
        'cell_phone_4', 'cereal_box_1', 'coffee_mug_3', 'comb_3', 'dry_battery_5', 'flashlight_2', 'food_bag_3',
        'food_box_5', 'food_can_9', 'food_cup_2', 'food_jar_1', 'garlic_4', 'glue_stick_3', 'greens_3', 'hand_towel_3',
        'instant_noodles_1', 'keyboard_5', 'kleenex_2', 'lemon_6', 'lightbulb_2', 'lime_3', 'marker_4', 'mushroom_1',
        'notebook_3', 'onion_4', 'orange_2', 'peach_2', 'pear_4', 'pitcher_2', 'plate_3', 'pliers_6', 'potato_3',
        'rubber_eraser_4', 'scissors_4', 'shampoo_5', 'soda_can_2', 'sponge_1', 'stapler_7', 'tomato_2', 'toothbrush_4',
        'toothpaste_5', 'water_bottle_1'
    ],
    [
        'apple_5', 'ball_3', 'banana_4', 'bell_pepper_1', 'binder_2', 'bowl_2', 'calculator_3', 'camera_2', 'cap_1',
        'cell_phone_3', 'cereal_box_1', 'coffee_mug_5', 'comb_2', 'dry_battery_3', 'flashlight_4', 'food_bag_8',
        'food_box_5', 'food_can_13', 'food_cup_1', 'food_jar_4', 'garlic_4', 'glue_stick_1', 'greens_1', 'hand_towel_3',
        'instant_noodles_6', 'keyboard_2', 'kleenex_5', 'lemon_2', 'lightbulb_1', 'lime_2', 'marker_2', 'mushroom_2',
        'notebook_1', 'onion_3', 'orange_3', 'peach_3', 'pear_8', 'pitcher_3', 'plate_7', 'pliers_6', 'potato_6',
        'rubber_eraser_3', 'scissors_2', 'shampoo_3', 'soda_can_2', 'sponge_5', 'stapler_4', 'tomato_2', 'toothbrush_1',
        'toothpaste_5', 'water_bottle_3'
    ],
    [
        'apple_3', 'ball_1', 'banana_2', 'bell_pepper_5', 'binder_1', 'bowl_6', 'calculator_4', 'camera_3', 'cap_4',
        'cell_phone_3', 'cereal_box_5', 'coffee_mug_5', 'comb_1', 'dry_battery_6', 'flashlight_3', 'food_bag_4',
        'food_box_6', 'food_can_3', 'food_cup_3', 'food_jar_2', 'garlic_5', 'glue_stick_1', 'greens_1', 'hand_towel_3',
        'instant_noodles_7', 'keyboard_1', 'kleenex_1', 'lemon_2', 'lightbulb_4', 'lime_2', 'marker_5', 'mushroom_1',
        'notebook_4', 'onion_6', 'orange_3', 'peach_2', 'pear_4', 'pitcher_1', 'plate_7', 'pliers_2', 'potato_4',
        'rubber_eraser_4', 'scissors_3', 'shampoo_3', 'soda_can_2', 'sponge_11', 'stapler_3', 'tomato_5',
        'toothbrush_3', 'toothpaste_2', 'water_bottle_7'
    ],
    [
        'apple_3', 'ball_1', 'banana_1', 'bell_pepper_4', 'binder_2', 'bowl_5', 'calculator_5', 'camera_3', 'cap_1',
        'cell_phone_1', 'cereal_box_1', 'coffee_mug_7', 'comb_3', 'dry_battery_4', 'flashlight_4', 'food_bag_3',
        'food_box_6', 'food_can_4', 'food_cup_3', 'food_jar_4', 'garlic_3', 'glue_stick_1', 'greens_4', 'hand_towel_3',
        'instant_noodles_4', 'keyboard_4', 'kleenex_4', 'lemon_4', 'lightbulb_1', 'lime_3', 'marker_4', 'mushroom_2',
        'notebook_3', 'onion_4', 'orange_2', 'peach_2', 'pear_2', 'pitcher_2', 'plate_2', 'pliers_6', 'potato_6',
        'rubber_eraser_1', 'scissors_1', 'shampoo_3', 'soda_can_2', 'sponge_11', 'stapler_1', 'tomato_4',
        'toothbrush_5', 'toothpaste_2', 'water_bottle_5'
    ]
 ]

with open("/local/home/luke/datasets/rgbd-dataset/test_splits.pckl", "wb") as f:
    pickle.dump(test_splits, f)