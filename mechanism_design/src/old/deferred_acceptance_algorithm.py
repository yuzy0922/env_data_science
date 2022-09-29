import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict

def create_data(num_suitors=10, num_reviewers=10, individual_rationality=True, outside_option="random"):
    suitors = [int(a) for a in list(range(1, num_suitors+1))]
    reviewers = [int(a) for a in list(range(num_suitors+1, num_suitors+num_reviewers+1))]
    
    suitor_prefs={}
    for s in suitors:
        suitor_prefs[s] = random.sample(reviewers, len(reviewers))
    reviewer_prefs={}
    for r in reviewers:
        reviewer_prefs[r] = random.sample(suitors, len(suitors))
        
    if individual_rationality:
        if outside_option=="random":
            for s in suitor_prefs:
                suitor_prefs[s].insert(random.choice(list(range(0, num_reviewers))), s)
            for r in reviewer_prefs:
                reviewer_prefs[r].insert(random.choice(list(range(0, num_suitors))), r)
        elif outside_option in range(0,num_reviewers):
            for s in suitor_prefs:
                suitor_prefs[s].insert(outside_option, s)
            for r in reviewer_prefs:
                reviewer_prefs[r].insert(outside_option, r)
    print("Prefs: ", suitor_prefs, reviewer_prefs)
    return suitor_prefs, reviewer_prefs

def suitor_without_match(matches, suitors):
    for suitor in suitors:
        if suitor not in matches:
            return suitor

def deferred_acceptance(suitor_prefs, reviewer_prefs, individual_rationality=True):
    all = list(suitor_prefs.keys()) + list(reviewer_prefs.keys())
    if not individual_rationality:
        reviewer_queue = defaultdict(int)
        suitors = list(suitor_prefs.keys())
        matches = {}
        while True:
            suitor = suitor_without_match(matches, suitors)
            if not suitor:
                break
            
            reviewer_index = reviewer_queue[suitor]
            reviewer_queue[suitor] += 1
    
            try:
                reviewer = suitor_prefs[suitor][reviewer_index]
            except IndexError:
                matches[suitor] = suitor
                continue
    
            print('Trying %s with %s... ' % (suitor, reviewer), end='')
            
            prev_suitor = matches.get(reviewer, None)
    
            if not prev_suitor:
                matches[suitor] = reviewer
                matches[reviewer] = suitor
                print('deferred')
                
            elif reviewer_prefs[reviewer].index(suitor) < \
                 reviewer_prefs[reviewer].index(prev_suitor):
                matches[suitor] = reviewer
                matches[reviewer] = suitor
                del matches[prev_suitor]
                print('matched')
                
            else:
                print('rejected')
                
        matched_pair = {suitor: matches[suitor] for suitor in suitor_prefs.keys()}
        for i in list(matched_pair.keys()):
            if matched_pair[i] == i:
                del matched_pair[i]
        matched_individuals = list(matched_pair.keys()) + list(matched_pair.values())
        unmatched_individuals = []
        for p in all:
            if p not in matched_individuals:
                unmatched_individuals.append(p)
        for u in unmatched_individuals:
            matches[u] = u
                
    if individual_rationality:
        reviewer_queue = defaultdict(int)
        suitors = list(suitor_prefs.keys())
        matches = {}
    
        while True:
            # Store a suitor from unmatched suitors
            suitor = suitor_without_match(matches, suitors)
            if not suitor:
                break
    
            # Store reviewer index
            reviewer_index = reviewer_queue[suitor]
            reviewer_queue[suitor] += 1
        
            try: 
                reviewer = suitor_prefs[suitor][reviewer_index]
            except IndexError:
                matches[suitor] = suitor
                continue
        
            suitor_self = suitor_prefs[suitor].index(suitor)
            if suitor_prefs[suitor].index(reviewer) < suitor_self:
                print('Trying %s with %s... ' % (suitor, reviewer), end='')
                prev_suitor = matches.get(reviewer, None)
                reviewer_self = reviewer_prefs[reviewer].index(reviewer)
                
                if not prev_suitor:
                    if reviewer_prefs[reviewer].index(suitor) < reviewer_self:
                        matches[suitor] = reviewer
                        matches[reviewer] = suitor
                        print('deferred')
                    elif reviewer_prefs[reviewer].index(suitor) > reviewer_self:
                        print("rejected")
                elif prev_suitor:
                    if reviewer_prefs[reviewer].index(suitor) < reviewer_prefs[reviewer].index(prev_suitor):
                        matches[suitor] = reviewer
                        matches[reviewer] = suitor
                        del matches[prev_suitor]
                        print('deferred')
                    elif reviewer_prefs[reviewer].index(suitor) > reviewer_prefs[reviewer].index(prev_suitor):
                        print("rejected")
                else:
                    print("rejected")
                    
            elif suitor_prefs[suitor].index(reviewer) > suitor_self:
                matches[suitor] = suitor
                del matches[suitor]
        
        matched_pair = {suitor: matches[suitor] for suitor in suitor_prefs.keys()}
        for i in list(matched_pair.keys()):
            if matched_pair[i] == i:
                del matched_pair[i]
        matched_individuals = list(matched_pair.keys()) + list(matched_pair.values())
        unmatched_individuals = []
        for p in all:
            if p not in matched_individuals:
                unmatched_individuals.append(p)
        for u in unmatched_individuals:
            matches[u] = u
    print("Matching result: ", {a: matches[a] for a in all})
    return {a: matches[a] for a in all}

def conduct_experiments(num_suitors, num_reviewers, outside_option, individual_rationality=True):
    suitor_prefs, reviewer_prefs = create_data(num_suitors=num_suitors, num_reviewers=num_reviewers, individual_rationality=individual_rationality, outside_option=outside_option)
   
    res = deferred_acceptance(suitor_prefs, reviewer_prefs, individual_rationality=individual_rationality)

    suitor_utils=[]
    for s in range(1, num_suitors+1):
        suitor_utils.append(num_suitors - suitor_prefs[s].index(res[s]))
    reviewer_utils=[]
    for r in range(num_suitors+1, num_suitors+num_reviewers+1):
        reviewer_utils.append(num_reviewers - reviewer_prefs[r].index(res[r]))
    sum_suitor_utils = np.sum(suitor_utils)
    sum_reviewer_utils = np.sum(reviewer_utils)
    print("Total utility of suitors: ", sum_suitor_utils)
    print("Total utility of reviewers: ", sum_reviewer_utils)
    return sum_suitor_utils, sum_reviewer_utils

def iterate_experiments(num_suitors, num_reviewers, num_itr, outside_option, individual_rationality=True):
    random.seed(20210721)
    sum_suitor_utils_list=[]
    sum_reviewer_utils_list=[]
    for itr in range(num_itr):
        sum_suitor_utils, sum_reviewer_utils = conduct_experiments(num_suitors,num_reviewers, individual_rationality, outside_option)
        sum_suitor_utils_list.append(sum_suitor_utils)
        sum_reviewer_utils_list.append(sum_reviewer_utils)
    return sum_suitor_utils_list, sum_reviewer_utils_list