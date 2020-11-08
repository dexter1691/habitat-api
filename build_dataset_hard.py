#!/usr/bin/env python
# coding: utf-8


import gzip
import json
import os
import sys
from typing import List
from matplotlib import pyplot as plt
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


# cd '/srv/flash1/hagrawal9/project/habitat/habitat-api/'

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

from rearrangement.utils.dataset import make_cfg, navmesh_settings, init_agent, init_episode_dict, add_goal_details, add_object_details, set_object_on_top_of_surface, settings, get_rotation
from rearrangement.utils.dataset import validate_again, validate_object, graph_validate, euclidean_distance, validate_object_goal_pointnav
from rearrangement.utils.visualization import get_top_down_map_sim
from rearrangement.utils.planner import compute_oracle_pickup_order_sim, compute_l2dist_pickup_order_sim, start_env_episode_distance_sim


object_templates = {}
with open('/srv/share3/hagrawal9/project/habitat/habitat-api/data/ycb_object_templates.json') as f:
    object_templates = json.load(f)
episodes = []

def init_test_scene_new(sim, simple_pathfinder, object_templates, navmesh_settings, num_objects, dist_thresh=5.0, object_obstacles=True):    
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
                og_geo_dist = geodesic_distance(simple_pathfinder, object_position, [goal_position])
                
                og_l2_dist = euclidean_distance(object_position, goal_position)
                og_dist_ratio = og_geo_dist / og_l2_dist

                if og_dist_ratio > 1.2 and goal_dist > 0.1 and ao_geo_dist <100:
                    # print(og_dist_ratio, og_geo_dist, og_l2_dist)
                    break
                
            if oi >= 100: 
                print('cannot find goal position')
                continue

            sim.set_object_motion_type(MotionType.DYNAMIC, object_id)
            sim.set_object_motion_type(MotionType.DYNAMIC, goal_id)
            
            
            sim.set_translation(object_position, object_id)
            sim.set_translation(goal_position, goal_id)
            sim.set_rotation(orientation, object_id)
            sim.set_rotation(orientation, goal_id)
            
            object_position = set_object_on_top_of_surface(sim, object_id)
            goal_position = set_object_on_top_of_surface(sim, goal_id)
            
            if not validate_object_goal_pointnav(sim, simple_pathfinder, agent_position, object_position, goal_position):
                # print("validate pointnav failed ")
                continue
            
            
            sim.set_object_motion_type(MotionType.STATIC, object_id)
            sim.set_object_motion_type(MotionType.STATIC, goal_id)
            
            if object_obstacles:
                sim.recompute_navmesh(
                    sim.pathfinder, 
                    navmesh_settings,
                    include_static_objects=True
                )
            
            if validate_object(
                sim, simple_pathfinder, agent_position, object_position, goal_position, 
                object_positions, goal_positions, 
                dist_threshold=dist_thresh
            ) and validate_again(sim, simple_pathfinder, object_positions, goal_positions):
                break

            
#             if graph_validate(
#                 sim, 
#                 simple_pathfinder,
#                 dist_threshold=dist_thresh
#             ):
# #                 print(
# #                     "added object: "
# #                     + str(object_id)
# #                     + " at: "
# #                     + str(object_position)
# #                 )
#                 break
            
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
        

def build_single_episode(sim, settings, scene_id, num_objects=5, object_obstacles=True):   
    simple_pathfinder = habitat_sim.PathFinder()
    name, ext = os.path.splitext(settings['scene'])
    simple_pathfinder.load_nav_mesh(name + ".navmesh")
    sim.recompute_navmesh(
        simple_pathfinder, navmesh_settings, False
    )

    sim.recompute_navmesh(sim.pathfinder, navmesh_settings, object_obstacles)
    
    episode = 0
    dist_thresh = 3.0

    object_count = num_objects
    episode_attempt_count = 0

    # clear the objects if we are re-running this initializer
    for old_obj_id in sim.get_existing_object_ids()[:]:
        sim.remove_object(old_obj_id)

    start_state, agent_object_id = init_agent(sim)
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings, object_obstacles)

    if num_objects == 5: 
        num_object = 5
    else: 
        num_object = np.random.choice(range(2, object_count + 1))

    object_positions, goal_positions, object_idxs, goal_idxs, object_template_idxs = init_test_scene_new(
        sim, simple_pathfinder, object_templates, navmesh_settings, num_object, dist_thresh, object_obstacles
    )

    result = graph_validate(sim, simple_pathfinder, agent_object_id, dist_thresh)
    
    if result == False or len(object_idxs) !=num_objects:
        # print("graph validation failed")
        episode_attempt_count += 1
        if episode_attempt_count % 20 == 0 and episode_attempt_count > 20: 
            print("Reducing object count")
            object_count -= 1
            object_count = max(2, object_count)
        return None , None

    episode_attempt_count = 0

    assert len(object_idxs) == len(goal_idxs)
    

    episode_dict = init_episode_dict(sim, settings['scene'], episode, agent_object_id)
    episode_dict = add_object_details(sim, episode_dict, len(object_idxs), object_idxs, object_template_idxs)
    episode_dict = add_goal_details(sim, episode_dict, len(object_idxs), goal_idxs)
    return episode_dict, agent_object_id


def sample_episode(sim, settings, scene_id, num_objects, threshold=0.95):
    episode, agent_object_id = build_single_episode(sim, settings, scene_id, num_objects=num_objects)

    if episode is None:
        return episode, agent_object_id, None, False
    
    i = 0
    j = 0
    while (i < 20 and j < 1000):
        object_positions = [obj['position'] for obj in episode['objects']]
        goal_positions = [obj['position'] for obj in episode['goals']]
        agent_pos = sim.get_agent(0).get_state().position
        top_down_map = None

        res = compute_oracle_pickup_order_sim(sim, sim.pathfinder, agent_pos, object_positions, goal_positions)
        res1 = compute_l2dist_pickup_order_sim(sim, agent_pos, object_positions, goal_positions)
        
        if res['pickup_order'] is None:
            pass
        elif res1['pickup_order_l2dist'] is None:
            pass
        else:
            dist = start_env_episode_distance_sim(sim, sim.pathfinder, agent_pos, object_positions, goal_positions, res['pickup_order'])        
            dist1 = start_env_episode_distance_sim(sim, sim.pathfinder, agent_pos, object_positions, goal_positions, res1['pickup_order_l2dist'])
            print('\r ratio: {:.3f} \tlength: {} \t i:{}'.format(dist/dist1, len(episodes)+1, i), end=" ")
            if (dist/dist1 < threshold):
                # print(res['pickup_order'], res1['pickup_order_l2dist'])
                episode['start_position'] = np.array(sim.agents[0].scene_node.translation).tolist()
                episode['start_rotation'] = get_rotation(sim, agent_object_id)

                return episode, agent_object_id, top_down_map, True
            i+=1
        
        sim.remove_object(agent_object_id)
        start_state, agent_object_id = init_agent(sim)
        j += 1

    return episode, agent_object_id, top_down_map, False


def main(args):
    split = args.split
    episode_num = args.episode_num
    num_objects  = args.num_object 

    train_df = pd.read_pickle('/srv/share3/hagrawal9/project/habitat/habitat-api/data/sokoban_gibson_{}.pkl'.format(args.split))
    
    print("Use object as obstacles: {}".format(args.object_obstacles))
    
    scene_id = train_df['id'].tolist()[args.start_idx]
    scene = 'data/scene_datasets/gibson_train_val/{}.glb'.format(scene_id)

    settings['scene'] = scene
    print("Scene: {}".format(scene))

    cfg = make_cfg(settings)
    sim = habitat_sim.Simulator(cfg)

    attempt = 0
    while(len(episodes) < episode_num):
        # episode = sample_episode(sim, settings, scene_id, num_objects)
        episode, agent_object_id, tdmap, s = sample_episode(sim, settings, scene_id, num_objects=5, threshold=args.threshold)
        
        if s:
            print("")
            episodes.append(episode)
            data = {
                'episodes': episodes, 
                'object_templates': object_templates
            }
            with gzip.open('/srv/flash1/hagrawal9/project/habitat/habitat-api/data/datasets/rearrangement/gibson/v1/{}/temp/rearrangement_hard_v7_{}_n={}_o={}_t={}_{}.json.gz'.format(
                split, split, len(episodes), num_objects, args.threshold, scene_id
            ), "wt") as f:
                json.dump(data, f)
                
        else:
            print('\n retrying {} \t attempt:{}'.format(len(episodes), attempt))
        attempt+=1

    sim.close()

    data = {
        'episodes': episodes, 
        'object_templates': object_templates
    }
    
    with gzip.open('/srv/flash1/hagrawal9/project/habitat/habitat-api/data/datasets/rearrangement/gibson/v1/{}/new/rearrangement_hard_v7_{}_n={}_o={}_t={}_{}.json.gz'.format(
        split, split, episode_num, num_objects, args.threshold, scene_id
    ), "wt") as f:
        json.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', dest='split', type=str, help='split')
    parser.add_argument('-s', dest='start_idx', type=int, help='start_idx')
    parser.add_argument('-n', dest='episode_num', type=int, help='episode_num')
    parser.add_argument('-o', dest='num_object', type=int, help='num_object')
    parser.add_argument('-t', dest='threshold', type=float, help="similarity threshold")
    parser.add_argument('--ignore-objects', dest="object_obstacles", action='store_false')
    args = parser.parse_args()
    main(args)
    
# CUDA_VISIBLE_DEVICES=1 GLOG_minloglevel=2 MAGNUM_LOG=quiet python build_dataset_hard.py -s 0 -n 100 -o 5 -d test -t 0.95