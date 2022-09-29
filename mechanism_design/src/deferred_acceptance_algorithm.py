import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def create_data(num_proposers=10, num_reviewers=10, outside_option="random", individual_rationality=True):
    proposers = [int(a) for a in list(range(1, num_proposers+1))]
    reviewers = [int(a) for a in list(range(num_proposers+1, num_proposers+num_reviewers+1))]
    
    proposer_prefs={}
    for p in proposers:
        proposer_prefs[p] = random.sample(reviewers, len(reviewers))
    reviewer_prefs={}
    for r in reviewers:
        reviewer_prefs[r] = random.sample(proposers, len(proposers))
        
    if individual_rationality:
        if outside_option=="random":
            for p in proposer_prefs:
                proposer_prefs[p].insert(random.choice(list(range(0, num_reviewers))), p)
            for r in reviewer_prefs:
                reviewer_prefs[r].insert(random.choice(list(range(0, num_proposers))), r)
        elif outside_option in range(0,num_reviewers+1):
            for p in proposer_prefs:
                proposer_prefs[p].insert(outside_option, p)
            for r in reviewer_prefs:
                reviewer_prefs[r].insert(outside_option, r)
    print(f"Prefs: , {proposer_prefs}, {reviewer_prefs}")
    return proposer_prefs, reviewer_prefs

def proposer_without_match(matches, proposers):
    for proposer in proposers:
        if proposer not in matches:
            return proposer

def deferred_acceptance(proposer_prefs, reviewer_prefs, individual_rationality=True):
    all = list(proposer_prefs.keys()) + list(reviewer_prefs.keys())
    if not individual_rationality:
        reviewer_queue = defaultdict(int)
        proposers = list(proposer_prefs.keys())
        matches = {}
        while True:
            proposer = proposer_without_match(matches, proposers)
            if not proposer:
                break
            
            reviewer_index = reviewer_queue[proposer]
            reviewer_queue[proposer] += 1
    
            try:
                reviewer = proposer_prefs[proposer][reviewer_index]
            except IndexError:
                matches[proposer] = proposer
                continue
    
            print(f'Trying {proposer} with {reviewer}... ' , end='')
            
            prev_proposer = matches.get(reviewer, None)
    
            if not prev_proposer:
                matches[proposer] = reviewer
                matches[reviewer] = proposer
                print('deferred')
                
            elif reviewer_prefs[reviewer].index(proposer) < \
                 reviewer_prefs[reviewer].index(prev_proposer):
                matches[proposer] = reviewer
                matches[reviewer] = proposer
                del matches[prev_proposer]
                print('matched')
                
            else:
                print('rejected')
                
        matched_pair = {proposer: matches[proposer] for proposer in proposer_prefs.keys()}
        for i in list(matched_pair.keys()):
            if matched_pair[i] == i:
                del matched_pair[i]
        matched_individuals = list(matched_pair.keys()) + list(matched_pair.values())
        unmatched_individuals = [p for p in all if p not in matched_individuals]
        for u in unmatched_individuals:
            matches[u] = u
                
    if individual_rationality:
        reviewer_queue = defaultdict(int)
        proposers = list(proposer_prefs.keys())
        matches = {}
    
        while True:
            # Store a proposer from unmatched proposers
            proposer = proposer_without_match(matches, proposers)
            if not proposer:
                break
    
            # Store reviewer index
            reviewer_index = reviewer_queue[proposer]
            reviewer_queue[proposer] += 1
        
            try: 
                reviewer = proposer_prefs[proposer][reviewer_index]
            except IndexError:
                matches[proposer] = proposer
                continue
        
            proposer_self = proposer_prefs[proposer].index(proposer)
            if proposer_prefs[proposer].index(reviewer) < proposer_self:
                print(f'Trying {proposer} with {reviewer}... ' , end='')
                prev_proposer = matches.get(reviewer, None)
                reviewer_self = reviewer_prefs[reviewer].index(reviewer)
                
                if not prev_proposer:
                    if reviewer_prefs[reviewer].index(proposer) < reviewer_self:
                        matches[proposer] = reviewer
                        matches[reviewer] = proposer
                        print('deferred')
                    elif reviewer_prefs[reviewer].index(proposer) > reviewer_self:
                        print("rejected")
                elif prev_proposer:
                    if reviewer_prefs[reviewer].index(proposer) < reviewer_prefs[reviewer].index(prev_proposer):
                        matches[proposer] = reviewer
                        matches[reviewer] = proposer
                        del matches[prev_proposer]
                        print('deferred')
                    elif reviewer_prefs[reviewer].index(proposer) > reviewer_prefs[reviewer].index(prev_proposer):
                        print("rejected")
                else:
                    print("rejected")
                    
            elif proposer_prefs[proposer].index(reviewer) > proposer_self:
                matches[proposer] = proposer
                del matches[proposer]
        
        matched_pair = {proposer: matches[proposer] for proposer in proposer_prefs.keys()}
        for i in list(matched_pair.keys()):
            if matched_pair[i] == i:
                del matched_pair[i]
        matched_individuals = list(matched_pair.keys()) + list(matched_pair.values())
        unmatched_individuals = [p for p in all if p not in matched_individuals]
        for u in unmatched_individuals:
            matches[u] = u
    print("Matching result: ", {a: matches[a] for a in all})
    return {a: matches[a] for a in all}

def conduct_experiments(num_proposers, num_reviewers, outside_option):
    proposer_prefs, reviewer_prefs = create_data(num_proposers=num_proposers, num_reviewers=num_reviewers, outside_option=outside_option)
   
    res = deferred_acceptance(proposer_prefs, reviewer_prefs)

    proposer_utils=[num_proposers - proposer_prefs[p].index(res[p]) for p in range(1, num_proposers+1)]
    reviewer_utils=[num_reviewers - reviewer_prefs[r].index(res[r]) for r in range(num_proposers+1, num_proposers+num_reviewers+1)]
    total_utils_proposers = np.sum(proposer_utils)
    total_utils_reviewers = np.sum(reviewer_utils)
    print(f"Total utility of proposers: , {total_utils_proposers}")
    print(f"Total utility of reviewers: , {total_utils_reviewers}")
    return proposer_prefs, reviewer_prefs, res, proposer_utils, reviewer_utils, total_utils_proposers, total_utils_reviewers

def iterate_experiments(num_proposers, num_reviewers, num_itr, outside_option):
    random.seed(20210721)

    proposer_prefs_list=[]
    reviewer_prefs_list=[]
    result_list=[]
    proposer_utils_list=[]
    reviewer_utils_list=[]
    total_utils_proposers_list=[]
    total_utils_reviewers_list=[]

    for itr in range(num_itr):
        proposer_prefs, reviewer_prefs, res, proposer_utils, reviewer_utils, total_utils_proposers, total_utils_reviewers = conduct_experiments(num_proposers,num_reviewers, outside_option)
        
        proposer_prefs_list.append(proposer_prefs)
        reviewer_prefs_list.append(reviewer_prefs)
        result_list.append(res)
        proposer_utils_list.append(proposer_utils)
        reviewer_utils_list.append(reviewer_utils)
        total_utils_proposers_list.append(total_utils_proposers)
        total_utils_reviewers_list.append(total_utils_reviewers)

    return proposer_prefs_list, reviewer_prefs_list, result_list, proposer_utils_list, reviewer_utils_list, total_utils_proposers_list, total_utils_reviewers_list