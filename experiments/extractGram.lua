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

require 'texture'
require 'image'
require 'optim'

require 'utils.DataLoader'
local utils = require 'utils.utils'
local preprocess = require 'utils.preprocess'
local opts = require 'opts'
local imgLoader = require 'utils.getImages'

local M = {}

function M.exec(opt)
	local styleLoader = imgLoader(opt.style_image_folder)
	if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]

  models = require('models/' .. opt.model)
	local cnet = models.createCNets(opt)
	
	cnet:cuda() 
	cnet:evaluate()

  local feat = {}
	for i = 1,styleLoader:size() do
		feat[i] = {}
		local style_image = styleLoader:get(i)
  	style_image = image.scale(style_image, opt.style_image_size)
  	style_image = preprocess.preprocess(style_image:add_dummy())
		feat[i] = cnet:calibrate(style_image:cuda())
	end

	local filename = opt.style_image_folder .. '/feat.t7'
  torch.save(filename, feat)
	print('Feats have been written to ', filename)
end

return M
