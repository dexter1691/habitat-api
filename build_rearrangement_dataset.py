import gzip
import json
import os
import sys
from typing import List

import argparse
import magnum as mn
import numpy as np
import pandas as pd
import networkx

import habitat_sim
from habitat_sim.physics import MotionType

from habitat.datasets.rearrangement.rearrangement_dataset import RearrangementDatasetV0
from habitat.tasks.rearrangement.rearrangement_task import RearrangementEpisode
from rearrangement.utils.geometry import geodesic_distance

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.default_agent_id = settings["default_agent_id"]
    sim_cfg.scene.id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    sim_cfg.physics_config_file = settings["physics_config_file"]

    # Note: all sensors must have the same resolution
    sensors = {
        "rgb": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "hfov": settings["hfov"]
        },
        "depth": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
            "hfov": settings["hfov"]
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.parameters["hfov"] = str(sensor_params["hfov"])

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

settings = {
    "max_frames": 10,
    "width": 640,  # Spatial resolution of the observations
    "height": 480,
    "hfov": 90,
    "scene": "data/scene_datasets/gibson_train_val/Barboursville.glb",  # Scene path
    "default_agent_id": 0,
    "sensor_height": 0.88,  # Height of sensors in meters
    "rgb": True,  # RGB sensor
    "depth": True,  # Depth sensor
    "seed": 1,
    "enable_physics": True,
    "physics_config_file": "data/default.phys_scene_config.json", 
    "silent": False, 
    "num_objects": 10,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "save_png": True
}

navmesh_settings = habitat_sim.NavMeshSettings()
navmesh_settings.set_defaults()
navmesh_settings.agent_radius = 0.2
navmesh_settings.agent_height = 0.88
navmesh_settings.agent_max_climb = 0.01

def register_object_templates(sim, object_templates):
    obj_attr_mgr = sim.get_object_template_manager()
    handles = obj_attr_mgr.get_file_template_handles()

    for sel_file_obj_handle, template_info in object_templates.items():
        obj_handle = os.path.basename(sel_file_obj_handle).split('.')[0]
        object_handle = obj_attr_mgr.get_file_template_handles(obj_handle)[0]

        obj_template = obj_attr_mgr.get_template_by_handle(object_handle)
        obj_template.scale = np.array(template_info['scale'])
        obj_attr_mgr.register_template(obj_template)

def init_agent(sim):
    obj_attr_mgr = sim.get_object_template_manager()
    handles = obj_attr_mgr.get_file_template_handles()
    object_handle = obj_attr_mgr.get_file_template_handles('sphere')[0]
    # Place the agent
    agent_pos = sim.pathfinder.get_random_navigable_point()
    # print(agent_pos)
    
    sim.agents[0].scene_node.translation = agent_pos
    
    agent_orientation_y = np.random.randint(0, 360)
    sim.agents[0].scene_node.rotation = mn.Quaternion.rotation(
        mn.Deg(agent_orientation_y), mn.Vector3(0, 1.0, 0)
    )
    
    agent_object_id = sim.add_object_by_handle(object_handle)
    sim.set_translation(agent_pos, agent_object_id)
    
    return sim.get_agent(0).get_state(), agent_object_id

def get_rotation(sim, oid):
    quat = sim.get_rotation(oid)
    return np.array(quat.vector).tolist() + [quat.scalar]

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(
        np.array(position_b) - np.array(position_a), ord=2
    )

def validate_object(sim, agent_position, object_position, goal_position, object_positions, goal_positions, dist_threshold=15.0):
    ao_geo_dist = geodesic_distance(sim.pathfinder, agent_position, [object_position])
    ag_geo_dist = geodesic_distance(sim.pathfinder, agent_position, [goal_position])
    og_geo_dist = geodesic_distance(sim.pathfinder, object_position, [goal_position])
    
    ao_l2_dist = euclidean_distance(agent_position, object_position)
    ag_l2_dist = euclidean_distance(agent_position, goal_position)
    og_l2_dist = euclidean_distance(object_position, goal_position)
    
    ao_dist_ratio = ao_geo_dist / ao_l2_dist
    og_dist_ratio = og_geo_dist / og_l2_dist
    
    if ao_l2_dist < 1.0 or ao_geo_dist > 100 or np.abs(object_position[1] - agent_position[1]) > 0.2:
        # print("ao:", ao_l2_dist, ao_geo_dist)
        return False
    
    if ag_geo_dist > 100 or np.abs(goal_position[1] - agent_position[1]) > 0.2:
        # print("ag:", ag_geo_dist)
        return False
    
    if og_l2_dist < dist_threshold or og_geo_dist > 100 or np.abs(object_position[1] - goal_position[1]) > 0.2:
        # print("og:", og_l2_dist, og_geo_dist)
        return False
    
    for j, curr_pos in enumerate([object_position, goal_position]):
        for i, pos in enumerate(object_positions + goal_positions):
            geo_dist = geodesic_distance(sim.pathfinder, curr_pos, [pos])
            l2_dist = euclidean_distance(curr_pos, pos)
            
            # check height difference to assure s and are from same floor
            if np.abs(curr_pos[1] - pos[1]) > 0.2: 
                
                return False
            
            if sim.pathfinder.island_radius(curr_pos) != sim.pathfinder.island_radius(pos):
                return False 
            
            if l2_dist < 0.5 or geo_dist > 100:
            
                return False

    return True
    
def validate_again(sim, object_positions, goal_positions):
    agent_position = sim.agents[0].scene_node.translation
    
    for i, posi in enumerate([agent_position] + object_positions + goal_positions):
        for j, posj in enumerate(object_positions + goal_positions):
            if np.abs(posi[1] - posj[1]) > 0.2:  # check height difference to assure s and t are from the same floor
                print("diff floor", i, j, posi, posj)
                return False

            geo_dist = geodesic_distance(sim.pathfinder, posi, [posj])

            if geo_dist > 100:
                print(geo_dist, i, j)
                return False

    return True

def graph_validate(sim, dist_threshold=15.0):
    
    grid_current_positions = []
    for sim_obj_id in sim.get_existing_object_ids():
        position = sim.get_translation(sim_obj_id)
        if sim_obj_id != agent_object_id:
            grid_current_positions.append(np.array(position))
        else:
            agent_position = np.array(position)
            
    dist_mat = np.zeros((   
            1 + len(grid_current_positions) , 
            1 + len(grid_current_positions)
    ))
    
    for i, posi in enumerate([agent_position] + grid_current_positions):
        for j, posj in enumerate([agent_position] + grid_current_positions):
            if i == j:
                continue
            
            geo_dist = geodesic_distance(sim, posi, [posj])
            l2_dist = euclidean_distance(posi, posj)
            
             # check height difference to assure s and are from same floor
            if np.abs(posi[1] - posj[1]) > 0.2: 
                # print("different height!")
                return False
            
            if l2_dist < 1.0:
                # print("distance between two objects less!", l2_dist, i, j, posi, posj)
                return False
            
            dist_mat[i, j] = geo_dist
    
    G = networkx.Graph(dist_mat)
    length = dict(networkx.algorithms.shortest_paths.all_pairs_shortest_path_length(G))
    
    for n1 in range(dist_mat.shape[0]):
        for n2 in range(dist_mat.shape[0]):
            if n2 not in length[n1]:
                # ipdb.set_trace()
                print(" not in dictionary")
                return False
            
    # ipdb.set_trace()
    # print(" validation true")
    return True

def init_episode_dict(sim, scene, episode_num, agent_object_id):
    episode_dict = {
        'episode_id': episode_num, 
        'scene_id': scene,
        'start_position': np.array(sim.agents[0].scene_node.translation).tolist(), 
        'start_rotation': get_rotation(sim, agent_object_id), 
        'info': {}, 
        'objects': [

        ],
        'goals': [ 
            
        ], 
    }
    return episode_dict

def add_object_details(sim, episode_dict, num_objects, object_idxs, object_template_idxs):
    for i in range(num_objects):
        obj_id = object_idxs[i]
        object_template = {
            'object_id': i, 
            'object_handle': object_template_idxs[i],
            'position': np.array(sim.get_translation(obj_id)).tolist(), 
            'rotation': get_rotation(sim, obj_id),
        }
        episode_dict['objects'].append(object_template)
    
    return episode_dict

def add_goal_details(sim, episode_dict, num_objects, goal_idxs):
    for i in range(num_objects):
        goal_id = goal_idxs[i]
        goal_template = {
            'position': np.array(sim.get_translation(goal_id)).tolist(), 
            'rotation': get_rotation(sim, goal_id),
        }
            
        episode_dict['goals'].append(goal_template)
    return episode_dict


def set_object_on_top_of_surface(sim, obj_id):
    r"""
    Adds an object in front of the agent at some distance.
    """

    obj_node = sim.get_object_scene_node(obj_id)
    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj_node.cumulative_bb, obj_node.transformation
    )

    # also account for collision margin of the scene
    scene_collision_margin = 0.00
    y_translation = mn.Vector3(
        0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
    )
    sim.set_translation(y_translation + sim.get_translation(obj_id), obj_id)
    
    return np.array(sim.get_translation(obj_id))

def init_test_scene_new(sim, object_templates, navmesh_settings, num_objects, dist_thresh=5.0, object_obstacles=True):    
    object_positions = []
    goal_positions = []
    object_idxs = []
    goal_idxs = []
    object_template_idxs = []
    
    obj_attr_mgr = sim.get_object_template_manager()
    agent_position = sim.agents[0].scene_node.translation
    
    sim.recompute_navmesh(
        sim.pathfinder, 
        navmesh_settings,
        include_static_objects=object_obstacles
    )
    
    for obj_id in range(num_objects):
        count = 0
        
        object_template_id = np.random.choice(list(object_templates.keys()))
        object_handle = obj_attr_mgr.get_file_template_handles(object_template_id)[0]
        rotation_x = mn.Quaternion.rotation(mn.Deg(-90), mn.Vector3(1.0, 0, 0))
        rotation_y = mn.Quaternion.rotation(mn.Deg(90), mn.Vector3(0.0, 1.0, 0))
        rotation_z = mn.Quaternion.rotation(mn.Deg(0), mn.Vector3(0.0, 0, 1.0))
        # rotation_x1 = mn.Quaternion.rotation(mn.Deg(-45), mn.Vector3(1.0, 0, 0))
        orientation = rotation_z * rotation_y * rotation_x

        object_id = sim.add_object_by_handle(object_handle)
        goal_id = sim.add_object_by_handle(object_handle)

        
        while count < 100:
            
            for oi in range(100):
                object_position = sim.pathfinder.get_random_navigable_point()
                # object_position[1] = agent_position[1]
                # set_object_on_top_of_surface(sim, object_id)
                
                object_dist = sim.pathfinder.distance_to_closest_obstacle(object_position, max_search_radius=2.0)
                ao_geo_dist = geodesic_distance(sim.pathfinder, agent_position, [object_position])
                
                if object_dist > 0.5 and ao_geo_dist < 100:
                    break

            if  oi >= 100: 
                continue 

             
            for oi in range(100):
                goal_position = sim.pathfinder.get_random_navigable_point()
                # goal_position[1] = agent_position[1]
                # set_object_on_top_of_surface(sim, goal_id)
                
                goal_dist = sim.pathfinder.distance_to_closest_obstacle(goal_position, max_search_radius=2.0)
                ao_geo_dist = geodesic_distance(sim.pathfinder, agent_position, [goal_position])
                if goal_dist > 0.5 and ao_geo_dist <100:
                    break

            if oi >= 100: 
                continue

            sim.set_object_motion_type(MotionType.DYNAMIC, object_id)
            sim.set_object_motion_type(MotionType.DYNAMIC, goal_id)
            
            
            # print("Object Dist: {}; Goal Dist: {}".format(object_dist, goal_dist))
            sim.set_translation(object_position, object_id)
            sim.set_translation(goal_position, goal_id)
            sim.set_rotation(orientation, object_id)
            sim.set_rotation(orientation, goal_id)
            # print(object_position, sim.get_translation(object_id))
            object_position = set_object_on_top_of_surface(sim, object_id)
            goal_position = set_object_on_top_of_surface(sim, goal_id)
            # print(object_position, sim.get_translation(object_id))
            
            sim.set_object_motion_type(MotionType.STATIC, object_id)
            sim.set_object_motion_type(MotionType.STATIC, goal_id)
            
            if object_obstacles:
                sim.recompute_navmesh(
                    sim.pathfinder, 
                    navmesh_settings,
                    include_static_objects=True
                )
            
#             if sim.pathfinder.is_navigable(object_position):
#                 print("init obj loc navigable")
#                 continue
#             if sim.pathfinder.is_navigable(goal_position):
#                 print("final obj loc navigable")
#                 continue
#             else:
#                 print("not navigable")
            
            if validate_object(
                sim, agent_position, object_position, goal_position, 
                object_positions, goal_positions, 
                dist_threshold=dist_thresh
            ) and validate_again(sim, object_positions, goal_positions):
#                 print(
#                     "added object: "
#                     + str(object_id)
#                     + " at: "
#                     + str(object_position)
#                 )
                break
            
            count += 1
        
        if count < 100:
            object_positions.append(object_position)
            goal_positions.append(goal_position)    
            object_idxs.append(object_id)
            goal_idxs.append(goal_id)
            object_template_idxs.append(object_template_id)
            # print("Success in {}".format(count))
        else:
            sim.remove_object(object_id)
            sim.remove_object(goal_id)
            return object_positions, goal_positions, object_idxs, goal_idxs, object_template_idxs

        if object_obstacles:
            # recompute navmesh so that objects don't overlap with other existing objects. 
            sim.recompute_navmesh(
                sim.pathfinder, 
                navmesh_settings,
                include_static_objects=True
            )
        
    return object_positions, goal_positions, object_idxs, goal_idxs, object_template_idxs
        
# set the number of objects to 1 always for now. 
def build_episode(train_df, episode_num, object_templates, num_objects=5, start_idx=0, end_idx=1, split="train", object_obstacles=True):
    global agent_object_id
    # for scene_id in train_df['id'].tolist()[5:10]:
    if end_idx == -1:
        end_idx = train_df['id'].shape[0]
        print(end_idx)

    for scene_id in train_df['id'].tolist()[start_idx: end_idx]:
        episodes = {'episodes': []}

        scene = 'data/scene_datasets/gibson_train_val/{}.glb'.format(scene_id)
        settings['scene'] = scene
        print(scene)
        
        cfg = make_cfg(settings)
        with habitat_sim.Simulator(cfg) as sim:
            sim.recompute_navmesh(sim.pathfinder, navmesh_settings, object_obstacles)
            # sv = SokobanVisualization(sim, navmesh_settings, map_resolution=(250, 250), num_samples=20000, draw_border=True)

            episode = 0
            dist_thresh = 3.0

            object_count = num_objects
            episode_attempt_count = 0
            while episode < episode_num:
                # clear the objects if we are re-running this initializer
                for old_obj_id in sim.get_existing_object_ids()[:]:
                    # print(old_obj_id)
                    sim.remove_object(old_obj_id)
                    
                start_state, agent_object_id = init_agent(sim)
                sim.recompute_navmesh(sim.pathfinder, navmesh_settings, object_obstacles)

                if num_objects == -1: 
                    num_object = 5
                else: 
                    num_object = np.random.choice(range(2, object_count + 1))
                
                num_object = 5
                object_positions, goal_positions, object_idxs, goal_idxs, object_template_idxs = init_test_scene_new(
                    sim, object_templates, navmesh_settings, num_object, dist_thresh, object_obstacles
                )

                sim.recompute_navmesh(sim.pathfinder, navmesh_settings, False)
                result = validate_again(sim, object_positions, goal_positions)
                if result == False or len(object_idxs) == 0:
                    episode_attempt_count += 1
                    if episode_attempt_count % 20 == 0 and episode_attempt_count > 20: 
                        print("Reducing object count")
                        object_count -= 1
                        object_count = max(2, object_count)
                    continue 

                episode_attempt_count = 0
                if num_objects == -1 and 5 != len(goal_idxs):
                    continue 
                     
                num_object = len(object_idxs)

                assert len(object_idxs) == len(goal_idxs)
                
                episode_dict = init_episode_dict(sim, scene, episode, agent_object_id)
                episode_dict = add_object_details(sim, episode_dict, num_object, object_idxs, object_template_idxs)
                episode_dict = add_goal_details(sim, episode_dict, num_object, goal_idxs)
                episodes['episodes'].append(episode_dict)
                print("\r Episode {} Object {}".format(episode, len(goal_idxs)), end=" ")
                episode += 1

            print("")
            episodes['object_templates'] = object_templates
            
            with gzip.open('/srv/flash1/hagrawal9/project/habitat/habitat-api/data/datasets/rearrangement/gibson/v1/{}/content/rearrangement_v3_{}_n={}_o={}_{}.json.gz'.format(
                split, split, episode_num, num_objects, scene_id
            ), "wt") as f:
                json.dump(episodes, f)
                
    return 


def main(args):
    train_df = pd.read_pickle('/srv/share3/hagrawal9/project/habitat/habitat-api/data/sokoban_gibson_{}.pkl'.format(args.split))
    object_templates = {}
    with open('/srv/share3/hagrawal9/project/habitat/habitat-api/data/ycb_object_templates.json') as f:
        object_templates = json.load(f)
    
    print("Use object as obstacles: {}".format(args.object_obstacles))
    build_episode(train_df, args.episode_num, object_templates, num_objects=args.num_object, start_idx=args.start_idx, end_idx=args.end_idx, split=args.split, object_obstacles=args.object_obstacles)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', dest='split', type=str, help='split')
    parser.add_argument('-s', dest='start_idx', type=int, help='start_idx')
    parser.add_argument('-e', dest='end_idx', type=int, help='end_idx')
    parser.add_argument('-n', dest='episode_num', type=int, help='episode_num')
    parser.add_argument('-o', dest='num_object', type=int, help='num_object')
    parser.add_argument('--ignore-objects', dest="object_obstacles", action='store_false')
    args = parser.parse_args()
    main(args)