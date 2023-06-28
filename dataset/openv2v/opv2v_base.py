import os
import os.path as osp
from collections import OrderedDict
from PIL import Image
from dataset.base_dataset import BaseDataset
from dataset.openv2v.utils import *


class OpV2VBase(BaseDataset):

    def __init__(self, cfgs, mode, use_cuda):
        super(OpV2VBase, self).__init__(cfgs, mode, use_cuda)
        self.proj_first = True
        self.max_box_num = 100
        if 'v2vreal' in self.cfgs['path'].lower():
            self.isSim = False
            self.order = 'hwl'
        else:
            self.isSim = True
            self.order = 'lwh'

    def init_dataset(self):
        # first load all paths of different scenarios
        scenario_folders = sorted(os.listdir(osp.join(self.cfgs['path'], self.mode)))
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.scenario_static_objects = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(scenario_folders):
            scenario_path = osp.join(self.cfgs['path'], self.mode, scenario_folder)
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_path)
                               if osp.isdir(
                    osp.join(scenario_path, x))])
            assert len(cav_list) > 0

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.cfgs['max_cav'] - 1:
                    # print('too many cavs')
                    break
                # load static objects
                static_obj_file = osp.join(scenario_path, 'static_vehicles.yaml')
                if os.path.exists(static_obj_file):
                    self.scenario_static_objects[i] = load_yaml(static_obj_file)

                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = osp.join(scenario_path, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([osp.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml')])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = osp.join(self.cfgs['path'], self.mode,
                                         scenario_folder, cav_id,
                                         timestamp + '.yaml')
                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file

                    lidar_file = osp.join(self.cfgs['path'], self.mode,
                                          scenario_folder, cav_id, timestamp
                                          + '_semantic_lidarcenter.bin')
                    if not osp.exists(lidar_file):
                        lidar_file = lidar_file.replace('_semantic_lidarcenter.bin', '.pcd')
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file

                    bevmap_file = osp.join(self.cfgs['path'], self.mode,
                                          scenario_folder, cav_id, timestamp
                                          + '_bev_road.png')
                    if osp.exists(bevmap_file):
                        self.scenario_database[i][cav_id][timestamp]['bevmap'] = \
                            bevmap_file

                    if self.cfgs['load_camera']:
                        camera_files = self.load_camera_files(cav_path, timestamp)
                        self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                            camera_files
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def __len__(self):
        return self.len_record[-1]

    def load_data(self, item):
        base_data_dict, scenario, timestamp_key = self.retrieve_base_data(item)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.cfgs['max_cav'])

        projected_lidar = []
        tf_matrices = []
        object_stack = []
        object_id_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            cav_pose = selected_cav_base['params']['lidar_pose']
            distance = dist_two_pose(cav_pose, ego_lidar_pose)
            if distance > self.cfgs['com_range']:
                continue

            selected_cav_processed = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)
            # only keep cavs that has enough lidar points in ego frame
            if len(selected_cav_processed['projected_lidar']) > 10:
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                projected_lidar.append(selected_cav_processed['projected_lidar'])
                tf_matrices.append(selected_cav_processed['transformation_matrix'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.max_box_num, 7))
        mask = np.zeros(self.max_box_num)
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        cav_num = len(projected_lidar)

        processed_data_dict['ego'].update(
            {'scenario': scenario,
             'timestamp': timestamp_key,
             'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'projected_lidar': projected_lidar,
             'tf_matrices': tf_matrices,
             'bev_map': base_data_dict[ego_id].get('bev_map', None),
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix})

        return processed_data_dict

    def retrieve_base_data(self, item):
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if item < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]
        # check the timestamp index
        timestamp_index = item if scenario_index == 0 else \
            item - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = \
                load_yaml(cav_content[timestamp_key]['yaml'])
            if scenario_index in self.scenario_static_objects:
                data[cav_id]['params']['vehicles'].update(
                    self.scenario_static_objects[scenario_index]
                )
            lidar_file = cav_content[timestamp_key]['lidar']
            if lidar_file[-3:] == 'bin':
                data[cav_id]['lidar_np'] = \
                    np.fromfile(lidar_file, dtype="float32").reshape(-1, 4)
            elif lidar_file[-3:] == 'pcd':
                data[cav_id]['lidar_np'] = \
                    np.loadtxt(lidar_file, skiprows=11).reshape(-1, 4)
            else:
                raise NotImplementedError
            if cav_content['ego'] and 'bev_map' in data[cav_id]:
                data[cav_id]['bev_map'] = \
                    Image.open(cav_content[timestamp_key
                               ]['bevmap']).__array__()[::-1, :]
            if self.cfgs['load_camera']:
                pass # todo: load camera

        scenario = list(scenario_database.values())[0][
            timestamp_key]['yaml'].split('/')[-3]
        return data, scenario, timestamp_key

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            generate_object_center([selected_cav_base],
                                   self.cfgs['lidar_range'],
                                   transformation_matrix if not self.isSim else ego_pose,
                                   self.order)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
               project_points_by_matrix_torch(lidar_np[:, :3],
                                              transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np, self.cfgs['lidar_range'])

        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'transformation_matrix': transformation_matrix,
             })

        return selected_cav_processed
    
    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = osp.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = osp.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = osp.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = osp.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix

    def map2points(self, map_mask, meters_per_pixel=0.2):
        """
        Transform map mask matrix to 2d points in ego lidar coordinates
        :param map_mask: np.ndarry(bool) [W, W]
        :return: np.ndarry(float) [N, 2], N=number of true elements in map_mask
        """
        points = np.stack(np.where(map_mask), axis=-1)
        points = (points - map_mask.shape[0] // 2)
        # points[:, 0] = - points[:, 0]
        points = points * meters_per_pixel

        return points

