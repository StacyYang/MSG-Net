--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2017
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

require 'nn'
require 'cunn'
require 'cudnn'

-- load packages from perceptual-loss 
require 'texture.utils'
require 'texture.GramMatrix'
require 'texture.ContentLoss'
require 'texture.StyleLoss'
require 'texture.TotalVariation'
require 'texture.PerceptualCriterion'
require 'texture.InstanceNormalization'

-- load MSG-Net dependencies
require 'texture.Calibrate'
require 'texture.Inspiration'
