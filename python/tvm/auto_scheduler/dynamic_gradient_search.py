# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Dynamic Gradient Descent Search algorithm in ICS'24: 
# import os
# from tvm.auto_scheduler.search_task import SearchTask

import numpy as np
import tvm
from tvm import te, auto_scheduler
from tvm.auto_scheduler.measure_record import load_records
import json
import json
from itertools import combinations, product
from math import isqrt

# def get_factors(n):
#     """
#     Return the factors of a given number n as a sorted list.
#     """
#     factors = set()
#     for i in range(1, isqrt(n) + 1):
#         if n % i == 0:
#             factors.add(i)
#             factors.add(n // i)
#     return sorted(factors)

# def extract_coordinates(record):
#     """
#     Extract coordinates from the SP nodes in the record.
#     """
#     json_str = json.loads(record)
#     idx_state = 1
#     coordinates = []
    
#     for each in json_str['i'][idx_state][1]:
#         if each[0] == "SP" and len(each[4]) == 1 and each[2] == 0:
#             continue
#         if each[0] == "SP":
#             coordinates.extend(each[4])
    
#     return coordinates

# def modify_sp_node(record, new_coordinates):
#     """
#     Modify the SP nodes in the record to match the new coordinates.
#     """
#     json_str = json.loads(record)
#     idx_state = 1
#     coord_idx = 0
    
#     for each in json_str['i'][idx_state][1]:
#         if each[0] == "SP" and len(each[4]) == 1 and each[2] == 0:
#             each[4] = [2]
#             continue
#         if each[0] == "SP":
#             length = len(each[4])
#             each[4] = new_coordinates[coord_idx:coord_idx + length]
#             coord_idx += length
    
#     return json.dumps(json_str)

class RecordProcessor:
    IDX_NODE_NAME = 0
    IDX_STAGE = 1
    IDX_ITER = 2
    IDX_LOOP_EXTENT = 3
    IDX_LENGTHS = 4
    IDX_INNER_TO_OUTER = 5
    IDX_TASK = 0
    IDX_STATE = 1

    def __init__(self, record):
        self.record = record
        self.json_str = json.loads(record)

    @staticmethod
    def get_factors(n):
        """
        Return the factors of a given number n as a sorted list.
        """
        factors = set()
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                factors.add(i)
                factors.add(n // i)
        return sorted(factors)

    def extract_coordinates(self):
        """
        Extract coordinates from the SP nodes in the record.
        """
        coordinates = []

        for each in self.json_str['i'][self.IDX_STATE][1]:
            if each[self.IDX_NODE_NAME] == "SP" and len(each[self.IDX_LENGTHS]) == 1 and each[self.IDX_ITER] == 0:
                continue
            if each[self.IDX_NODE_NAME] == "SP":
                coordinates.extend(each[self.IDX_LENGTHS])

        return coordinates

    def modify_sp_node(self, new_coordinates):
        """
        Modify the SP nodes in the record to match the new coordinates.
        """
        coord_idx = 0

        for each in self.json_str['i'][self.IDX_STATE][1]:
            if each[self.IDX_NODE_NAME] == "SP" and len(each[self.IDX_LENGTHS]) == 1 and each[self.IDX_ITER] == 0:
                # # if loop extend is the multiple of 2, modify to [2]
                # if each[self.IDX_LOOP_EXTENT] % 2 == 0:
                #     each[self.IDX_LENGTHS] = [2]
                continue
            if each[self.IDX_NODE_NAME] == "SP":
                length = len(each[self.IDX_LENGTHS])
                each[self.IDX_LENGTHS] = new_coordinates[coord_idx:coord_idx + length]
                coord_idx += length
            if each[self.IDX_NODE_NAME] == "PR":
                # aggresive unroll
                each[self.IDX_LOOP_EXTENT] = "auto_unroll_max_step$1024"

        self.record = json.dumps(self.json_str)

class DynamicGradientSearchTuner:
    def __init__(self, task, log_file, n_trials=5, init_size=64, slide_window_size=3):
        """
        Initialize the DynamicGradientSearch object.

        Parameters:
        - task: The task to be optimized.
        - log_file: The file path to save the optimization log.
        - n_trials: The number of trials to perform during optimization.
        - init_size: The initial size of the model
        - slide_window_size: The size of the sliding window used for gradient descent.

        Returns:
        None
        """
        self.task = task
        self.log_file = log_file
        self.n_trials = n_trials
        self.init_size = init_size
        self.slide_window_size = slide_window_size
        self.model = auto_scheduler.XGBModel(num_warmup_sample=1)
        self.measured_throughputs_ = []
        self.count_total_measured = 0
        self.visited = set()
        self.isCUDA = False
        self.max_trials = 1000

    def get_sample_records(self, log_file, number, task):
        """Generate a list of random MeasureInput and MeasureResult pairs"""

        policy = auto_scheduler.SketchPolicy(task, program_cost_model=self.model, verbose=0)
        states = policy.sample_initial_population()
        states = states[:min(number, len(states))]
        
        self.count_total_measured += len(states)

        inputs = [auto_scheduler.MeasureInput(task, s) for s in states]
        
        local_builder = auto_scheduler.LocalBuilder()
        local_runner = auto_scheduler.LocalRunner(timeout=10)

        bress = local_builder.build(inputs)
        # assert bress[0].error_no == 0
        mress = local_runner.run(inputs, bress)
        del local_builder
        del local_runner
        
        # assert mress[0].error_no == 0
        
        # for inp, res in zip(inputs, mress):
        #     record_str = auto_scheduler.measure_record.dump_record_to_string(inp, res)
        #     print("Record: ", record_str)
        
        # # use the first input and result to get the record string as a template for mutation and get neighbors
        # record_str = auto_scheduler.measure_record.dump_record_to_string(inputs[0], mress[0])
        # print("Record: ", record_str)
        
        with open(log_file, "a") as fp:
            auto_scheduler.save_records(fp.name, inputs, mress)

        return task, inputs, mress


    def DGD_Search(self, log_file, record, task, slide_window_size = 3):
        measured_inputs = []
        measured_results = []
        base_input, base_result = auto_scheduler.measure_record.load_record_from_string(record)
        
        states_1hop_record = self.get_n_hop_neighbors(record, 1)
        states_2hop_record = self.get_n_hop_neighbors(record, 2)
        
        all_neighbors = states_1hop_record + states_2hop_record
        
        # print(f"debug >> size of all_neighbors: {len(all_neighbors)}")
        
        candidate_inputs = [base_input]
        for record_str in all_neighbors:
            # get all 1+2 hops and predict/sorted by scores
            inp, _ = auto_scheduler.measure_record.load_record_from_string(record_str)
            candidate_inputs.append(inp)
        
        candidate_scores = self.model.predict(task, [x.state for x in candidate_inputs])
        base_score = candidate_scores[0]
        candidate_scores = candidate_scores[1:]
        candidate_inputs = candidate_inputs[1:]
        
        # move to the next base
        new_base, tmp_measured_inputs, tmp_measured_results = self.DGD_Move(log_file, base_result, base_score, candidate_inputs, candidate_scores, slide_window_size)\
        
        if self.count_total_measured >= self.max_trials:
            return new_base, measured_inputs, measured_results
        
        measured_inputs.extend(tmp_measured_inputs)
        measured_results.extend(tmp_measured_results)
        
        if not new_base:
            # didn't find new base, then explore 3hop for the current base
            print(">>>> explore 3hop")
            all_neighbors = self.get_n_hop_neighbors(record, 3)
            # print(f"debug >> size of states_3hop_record: {len(all_neighbors)}")
            
            candidate_inputs = [base_input]
            for record_str in all_neighbors:
                # get all 3 hops and predict/sorted by scores
                inp, _ = auto_scheduler.measure_record.load_record_from_string(record_str)
                candidate_inputs.append(inp)
                
            candidate_scores = self.model.predict(task, [x.state for x in candidate_inputs])
            base_score = candidate_scores[0]
            
            candidate_scores = candidate_scores[1:]
            candidate_inputs = candidate_inputs[1:]
            
            new_base, tmp_measured_inputs, tmp_measured_results = self.DGD_Move(log_file, base_result, base_score, candidate_inputs, candidate_scores, slide_window_size)
            
            if self.count_total_measured >= self.max_trials:
                return new_base, measured_inputs, measured_results
            
            measured_inputs.extend(tmp_measured_inputs)
            measured_results.extend(tmp_measured_results)
                
        return new_base, measured_inputs, measured_results

    def DGD_Move(self, log_file, base_result, base_score, candidate_inputs, candidate_scores, slide_window_size):
        assert len(candidate_inputs) == len(candidate_scores)
        
        score_threshold = base_score * 0.6
        base_cost = np.mean([v.value for v in base_result.costs])
        global measured_throughputs_
        measured_throughputs_.append(1/base_cost)
        
        print(f"base_cost: {base_cost}")
        # sort from large to small
        sorted_indices = np.argsort(candidate_scores)[::-1]
        
        # Skip candidates with score lower than score threshold
        sorted_indices = [idx for idx in sorted_indices if candidate_scores[idx] >= score_threshold]
        print("number of candidates after scores filtering: ", len(sorted_indices))
        
        next_base = None
        measured_inputs = []
        measured_results = []
        
        # apply slide window to the sorted indices, and measure the slide window, until find a better cost neighbor,
        index_slide = 0
        
        while index_slide < len(sorted_indices) and not next_base:
            
            if index_slide + slide_window_size > len(sorted_indices):
                slide_window_indices = sorted_indices[index_slide:]
            else: # slide_window_size <= len(sorted_indices)
                slide_window_indices = sorted_indices[index_slide:index_slide+slide_window_size]
            
            slide_window_scores = [candidate_scores[i] for i in slide_window_indices]
            print(f"slide_window_scores: {slide_window_scores}")
            
            # get the slide window inputs
            slide_window_inputs = [candidate_inputs[i] for i in slide_window_indices]
            
            # measure the slide window inputs
            local_builder = auto_scheduler.LocalBuilder()
            local_runner = auto_scheduler.LocalRunner(timeout=10)
            bress = local_builder.build(slide_window_inputs)
            slide_window_results = local_runner.run(slide_window_inputs, bress)
            
            del local_builder
            del local_runner
            
            slide_window_costs = []
            for res in slide_window_results:
                slide_window_costs.append(np.mean([v.value for v in res.costs]))
            print(f"slide_window_costs: {slide_window_costs}")
            
            
            # break after self.max_trials measurements
            if self.count_total_measured + len(slide_window_inputs) >= self.max_trials:
                # need to save to the log_file
                tmp_size = min(len(slide_window_inputs), self.max_trials - self.count_total_measured)
                with open(log_file, "a") as fp:
                    tmp_inputs = slide_window_inputs[:tmp_size]
                    tmp_results = slide_window_results[:tmp_size]
                    
                    auto_scheduler.save_records(fp.name, tmp_inputs, tmp_results)
                    
                self.count_total_measured += tmp_size
                
                print("count_total_measured: ", self.count_total_measured)
                print(">>>>>>>>>>>>>>>> Done DGD_Search <<<<<<<<<<<<<<<<")
                return next_base, measured_inputs, measured_results
            
            # used for budget control
            self.count_total_measured += len(slide_window_inputs)
            
            # need to save to the log_file
            with open(log_file, "a") as fp:
                auto_scheduler.save_records(fp.name, slide_window_inputs, slide_window_results)
            
            # print(f"slide_window_results: {slide_window_results}")
                        
            index_slide += slide_window_size
            # used for updating the model
            measured_inputs.extend(slide_window_inputs)
            measured_results.extend(slide_window_results)
            
            # add to measured_throughputs_
            for cost in slide_window_costs:
                measured_throughputs_.append(1/cost)
            
            # threshold
            best_measured = np.max(measured_throughputs_)
            measure_threshold = best_measured * 0.6
            
            # early stop
            if 1/np.min(slide_window_costs) < measure_threshold and index_slide > 3*slide_window_size:
                print(f"early stop: best in slide_window={1/np.min(slide_window_costs)} < measure_threshold={measure_threshold}")
                break
            
            # # if the best cost in the slide window is better than the base cost, then use it as the next base
            # if np.min(slide_window_costs) < base_cost:
            #     print(f">>> find a better cost in the slide window")
            #     print(f"current base cost: {base_cost}")
            #     print(f"new base cost: {np.min(slide_window_costs)}")
            #     next_base_inp = slide_window_inputs[np.argmin(slide_window_costs)]
            #     next_base_res = slide_window_results[np.argmin(slide_window_costs)]
            #     next_base = auto_scheduler.measure_record.dump_record_to_string(next_base_inp, next_base_res)
            #     print("new base: ", next_base)
            #     break
            sorted_idx = np.argsort(slide_window_costs)
            # find a better cost to move, add to visited, and avoid re-visit
            for idx in sorted_idx:
                if slide_window_costs[idx] < base_cost and slide_window_inputs[idx] not in self.visited:
                    next_base_inp = slide_window_inputs[idx]
                    next_base_res = slide_window_results[idx]
                    next_base = auto_scheduler.measure_record.dump_record_to_string(next_base_inp, next_base_res)
                    print("new base: ", next_base)
                    # add to visited
                    self.visited.add(next_base_inp)
                    break
            
        return next_base, measured_inputs, measured_results

    def get_n_hop_neighbors(self, record, n):
        """
        Generate n-hop neighbors for the given record.
        """
        processor = RecordProcessor(record)
        original_coordinates = processor.extract_coordinates()
        dimension = len(original_coordinates)
        neighbors = []

        # Generate all combinations of coordinates to change
        for indices in combinations(range(dimension), n):
            # Generate all possible changes for the selected coordinates
            for changes in product([-1, 1], repeat=n):
                new_coordinates = original_coordinates[:]
                coord_idx = 0
                valid_change = True  # Add a flag to ensure changes are valid
                for each in processor.json_str['i'][processor.IDX_STATE][1]:
                    if each[processor.IDX_NODE_NAME] == "SP" and len(each[processor.IDX_LENGTHS]) == 1 and each[processor.IDX_ITER] == 0:
                        continue
                    if each[processor.IDX_NODE_NAME] == "SP":
                        length = len(each[processor.IDX_LENGTHS])
                        dim_len = each[processor.IDX_LOOP_EXTENT]
                        factors = processor.get_factors(dim_len)
                        for i, change in enumerate(changes):
                            idx = indices[i]
                            if coord_idx <= idx < coord_idx + length:
                                current_value = new_coordinates[idx]
                                if current_value in factors:
                                    factor_index = factors.index(current_value)
                                    new_factor_index = factor_index + change
                                    if 0 <= new_factor_index < len(factors):
                                        new_coordinates[idx] = factors[new_factor_index]
                                    else:
                                        valid_change = False
                                        break
                                else:
                                    valid_change = False
                                    break
                                if valid_change:
                                    if self.isCUDA and new_coordinates[coord_idx] != 1 and length >= 3:
                                        # Force the cuda code has no vthread on parallel dimensions
                                        valid_change = False
                                        break
                        if valid_change:
                            product_of_dims = np.prod(new_coordinates[coord_idx:coord_idx + length])
                            if product_of_dims > dim_len:
                                valid_change = False
                                break
                        coord_idx += length
                if valid_change and new_coordinates != original_coordinates:
                    modified_processor = RecordProcessor(json.dumps(processor.json_str))
                    modified_processor.modify_sp_node(new_coordinates)
                    neighbors.append(modified_processor.record)
        
        return neighbors
    
    # def get_n_hop_neighbors(self, record, n):
    #     """
    #     Generate n-hop neighbors for the given record.
    #     """
    #     json_str = json.loads(record)
    #     idx_state = 1
    #     original_coordinates = extract_coordinates(record)
    #     dimension = len(original_coordinates)
    #     neighbors = []

    #     # Generate all combinations of coordinates to change
    #     for indices in combinations(range(dimension), n):
    #         # Generate all possible changes for the selected coordinates
    #         for changes in product([-1, 1], repeat=n):
    #             new_coordinates = original_coordinates[:]
    #             coord_idx = 0
    #             valid_change = True  # Add a flag to ensure changes are valid
    #             for each in json_str['i'][idx_state][1]:
    #                 if each[0] == "SP":
    #                     length = len(each[4])
    #                     dim_len = each[3]
    #                     factors = get_factors(dim_len)
    #                     for i, change in enumerate(changes):
    #                         idx = indices[i]
    #                         if coord_idx <= idx < coord_idx + length:
    #                             current_value = new_coordinates[idx]
    #                             if current_value in factors:
    #                                 factor_index = factors.index(current_value)
    #                                 new_factor_index = factor_index + change
    #                                 if 0 <= new_factor_index < len(factors):
    #                                     new_coordinates[idx] = factors[new_factor_index]
    #                                 else:
    #                                     valid_change = False
    #                                     break
    #                             else:
    #                                 valid_change = False
    #                                 break
    #                     if valid_change:
    #                         product_of_dims = np.prod(new_coordinates[coord_idx:coord_idx + length])
    #                         if product_of_dims > dim_len:
    #                             valid_change = False
    #                             break
    #                     coord_idx += length
    #             if valid_change and new_coordinates != original_coordinates:
    #                 modified_record = modify_sp_node(record, new_coordinates)
    #                 neighbors.append(modified_record)
        
    #     return neighbors

    # def get_nhop(self, record, n):
    #     """
    #     Get the nhop for a given record.
        
    #     Split Node structure: ("SP", stage_id, iter_id, loop_extent, lengths[], inner_to_outer)
        
    #     stage_id: The id of the stage
    #     iter_id: The id of the loop iteration
    #     loop_extent: Problem size
    #     lengths: Loop tiling sizes
        
    #     Parameters:
    #     - record: str
    #         The record to process.
    #     - n: int
    #         The number of hops to calculate.

    #     Returns:
    #     - list
    #         A list containing the nhop for the given record.
    #     """
    #     # for SplitNode
    #     idx_Node_name, idx_stage, idx_iter, idx_loop_extent, idx_lengths, idx_inner_to_outer = 0, 1, 2, 3, 4, 5
    #     # for MeasureInput(task, state)
    #     idx_task, idx_state = 0, 1
        
    #     print("getting nhop for record: ", record)
        
    #     json_str = json.loads(record)
    #     print(f"json format: {json.dumps(json_str, indent=2)}")
    #     print(f"non-json format: {json_str}")
    #     print(f"-------------------")
        
    #     # find the first SP node in the state, and change the lengths to 1, 2, 3, 4
    #     total_lenghts = 0
    #     for ite, each in enumerate(json_str['i'][idx_state][1]):
    #         print(f"ite: {ite}, each: {each}")
    #         # if each[idx_Node_name] == "SP" and len(each[idx_lengths]) != 1 and each[idx_iter] != 0:
    #         if each[idx_Node_name] == "SP" and len(each[idx_lengths]) != 1:
    #             print(f"ite: {ite}, each: {each}")
    #             print(f"each[idx_lengths]: {each[idx_lengths]}")
    #             each[idx_lengths] = [1, 1, 1]
    #             print(f"new each[idx_lengths]: {each[idx_lengths]}")
    #             break
    #         # if each[idx_Node_name] == "SP" and len(each[idx_lengths]) != 1 and each[idx_iter] != 0:
    #         #     total_lenghts += len(each[idx_lengths])
    #     # print(f"json format: {json.dumps(json_str, indent=2)}")
    #     print(f"new non-json format: {json_str}")
    #     print("total_lenghts: ", total_lenghts)
    #     input("Press Enter to continue...")
        
    #     ############# testing use MeasureInput
    #     # inp, res = auto_scheduler.measure_record.load_record_from_string(record)
    #     # input_str = auto_scheduler._ffi_api.SerializeMeasureInput(inp)
    #     # print("input_str: ", input_str)
    #     # print("back_to_inp: ", back_to_inp)
    #     # back_to_inp = auto_scheduler.measure.recover_measure_input(back_to_inp)
    #     # back_to_inp_str = auto_scheduler._ffi_api.DeserializeMeasureInput(back_to_inp)
    #     # print(f"back_to_inp_str: {back_to_inp_str}")
    #     # assert back_to_inp == inp
    #     # print("inp.state: ", inp.state)
    #     # print("str(inp.state): ", str(inp.state))
    #     # print("res.costs: ", res.costs)
    #     # print()
    #     ############# testing use MeasureInput end
        
    #     # print elements in the json_str
    #     for key in json_str:
    #         print(f">> key: {key}")
    #         print(f"value: {json_str[key]}")
    #         if key == 'i':
    #             for ite, each in enumerate(json_str[key]):
    #                 print(f"-- ite: {ite}, each: {each}")

    #     # input("Press Enter to continue...")
        
    #     return [record]

    def dynamic_gradient_search(self): #, log_file, task, init_size = 64, n_trials = 5, slide_window_size = 3):
        log_file = self.log_file
        task = self.task
        init_size = self.init_size
        n_trials = self.n_trials
        slide_window_size = self.slide_window_size
        
        count_total_measured = init_size
        if "cuda" in str(task.target):
            print(">>>>>>>>>>>>>>>> Start DGD_Search for CUDA <<<<<<<<<<<<<<<<")
            print("apply DGD space and optimization")
            self.isCUDA = True
            task.hardware_params.max_vthread_extent = 1
        else:
            print(">>>>>>>>>>>>>>>> Start DGD_Search for CPU <<<<<<<<<<<<<<<<")
            
        # use 1/exe_time as the throughput
        global measured_throughputs_
        measured_throughputs_ = []
        # topk = n_trials
        # # get the top k records
        # list_costs = []
        # records = []
        # # load the records from the log file
        # for inp, res in load_records(log_file):
        #     # print("type res: ", type(res))
        #     costs = [v.value for v in res.costs]
        #     cost = np.mean(costs)
        #     list_costs.append(cost)
        #     record_str = auto_scheduler.measure_record.dump_record_to_string(inp, res)
        #     records.append(record_str)
            
        # topk_indices = np.argsort(list_costs)[:topk]
        # topk_records = [records[i] for i in topk_indices]
        # for record in topk_records:
        #     print("record: ", record)

        task, inputs, results = self.get_sample_records(log_file, init_size, task)
        
        self.model.update(inputs, results)
        
        list_costs = []
        records = []
        topk = n_trials
        for inp, res in zip(inputs, results):
            record_str = auto_scheduler.measure_record.dump_record_to_string(inp, res)
            costs = [v.value for v in res.costs]
            cost = np.mean(costs)
            list_costs.append(cost)
            records.append(record_str)

        topk_indices = np.argsort(list_costs)[:topk]
        topk_records = [records[i] for i in topk_indices]
        assert len(topk_records) == n_trials
        
        # use topk as budget now, later will add more options like ntrials budget
        for ite, record in enumerate(topk_records):
            print('size of topk_records: ', len(topk_records))
            print("ite: ", ite)
            while record != None:
                print("current base: ", record)
                
                # print(f"base_input: {base_input}")
                # print(f"base_input.state: {base_input.state}")
                # print(f"base_input.task: {base_input.task}")
                # print(f"base_input.task.compute_dag: {base_input.task.compute_dag}")
                
                # task_record = auto_scheduler._ffi_api.SerializeSearchTask(base_input.task)
                # print(f"task_record: {task_record}")
                # new_task = auto_scheduler._ffi_api.DeserializeSearchTask(task_record)
                # print(f"new_task: {new_task}")

                # print(f"base_result: {base_result}")
                # # input("Press Enter to continue...")
                
                # all_neighbors = DGD_Search(record, 2)
                # print(f"debug>> size of all_neighbors: {len(all_neighbors)}")
                # neighbors_inputs = []
                # for record_str in all_neighbors:
                #     # get all 1+2 hops and predict/sorted by scores

                #     inp, _ = auto_scheduler.measure_record.load_record_from_string(record_str)
                #     neighbors_inputs.append(inp)
                
                # candidate_scores = self.model.predict(task, [x.state for x in neighbors_inputs])
                # print(f"candidate_scores: {candidate_scores}")
                
                # # move to the next base
                # record, measured_inputs, measured_results = DGD_Move(log_file, base_result, neighbors_inputs, candidate_scores, slide_window_size)
                
                record, measured_inputs, measured_results = self.DGD_Search(log_file, record, task, slide_window_size)
                
                if self.count_total_measured >= self.max_trials:
                    return 
                
                # update the model with the new results
                self.model.update(measured_inputs, measured_results)
                