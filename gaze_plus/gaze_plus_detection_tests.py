import h5py

GAZEPLUS_DIR = "detections/"
TEST_ACTORS = ["Ahmad", "Alireza", "Carlos", "Rahul"]

FILENAME_ROOT = "fasterrcnn_output_final_"

ACTOR_IN_TEST = "Ahmad" # it has to be in TEST_ACTORS

print("You selected {} as the actor in test".format(ACTOR_IN_TEST))

# Open one of the files representing a whole fold (it depens on actor in test)
h5 = h5py.File(GAZEPLUS_DIR + FILENAME_ROOT + ACTOR_IN_TEST + ".h5", 'r')

# Mode between ‘train’, ‘val’, and ‘test’; ‘actor_in_test’ is used to determine which training fold is used, where ‘Ahmad’ for example in test would mean that the rest of actors are in the train set
mode = "train"
data_set = h5['{}_{}'.format(mode, ACTOR_IN_TEST)]
print("Actors in this {} fold: {}".format(mode, data_set.keys()))

for video_actor_name in data_set.keys():    
    # Choose the actor whose videos are going to be obtained
    video_actor = data_set[video_actor_name]
    print("Action videos for actor {}: {}".format(video_actor_name, video_actor.keys()))

    for video_actions in video_actor.keys():        
        # Choose the class (name of it, as it appears on the class folders)
        video_action = video_actor[video_actions]
        print("Number of videos for action {}: {}".format(video_actions, len(video_action.keys())))
        for video_instance in video_action.keys():
            # For each video, we have ['bounding_boxes', 'fc', 'nb_detections', 'object_label', 'scores']            
            # Obtain data from a specific instance (video)
            video = video_action[video_instance]
            print("Video information: {}".format(video.keys()))
            # shape bounding_boxes == (nb_frames, 10, 4) -> coordinates y1,x1,y2,x2
            bounding_boxes = video['bounding_boxes']
            print("Bounding boxes shape: {}".format(bounding_boxes.shape))
            
            

