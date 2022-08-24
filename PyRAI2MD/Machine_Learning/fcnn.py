#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:08:14 2022

@author: Chen Zhou
"""
import pickle
import os
import time
import numpy as np

class FCNN(object):
    def __init__(self, keywords = None, id = None):
        self.metadata_dir = "/home/chen/Documents/BlueOLED/NNsForMD/blueOLED/pyrai2md/metadata"
        self.natom = 38
        self.nstate = 2
        
    def train(self):
        ## fake	function does nothing
        pass
    
    def load(self):
        ## fake	function does nothing
        pass
    
    def	appendix(self,addons):
       	## fake	function does nothing
        pass
    
    def evaluate(self, traj):
        # write coordinates for ML model to read
        coord = traj.coord.reshape((1, self.natom, 3))[0]
        xyz_file = os.path.join(self.metadata_dir, "xyz.dat")
        with open(xyz_file, 'wb') as fh:
            np.save(fh, coord)
        
        # indicate the start of prediction by create run_ml file
        start_file = os.path.join(self.metadata_dir, "run_ml")
        fh = open(start_file, 'w')
        fh.close()
        
        res_file = os.path.join(self.metadata_dir, "pred_res.npy")
        while(os.path.isfile(start_file)):
            print("Wait for prediction...")
            # wait for prediction to be finished
            time.sleep(0.1)
        
        print("Prediction received. MD processing...")
        with open(res_file, 'rb') as fh:
            energy = np.load(fh)
            gradient = np.load(fh)
            nac = np.load(fh)
            soc = np.load(fh)
            err_e = np.load(fh)
            err_g = np.load(fh)
            err_n = np.load(fh)
            err_s = np.load(fh)
            #energy, gradient, nac, soc, err_e, err_g, err_n, err_s = pickle.load(fh)
        traj.energy = np.copy(energy)
        traj.grad = np.copy(gradient)
        traj.nac = np.copy(nac)
        traj.soc = np.copy(soc)
        traj.err_energy = err_e
        traj.err_grad = err_g
        traj.err_nac = err_n
        traj.err_soc = err_s
        traj.status = 1
        
        # remove res_file to avoid conflict with the next run
        os.system('rm -f %s' % res_file)
        
        return traj
