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

local pad = nn.SpatialReflectionPadding
local normalization = nn.InstanceNormalization
local layer_utils = require 'texture.layer_utils'

local M = {}

function M.createModel(opt)
	-- Global variable keeping track of the input channels
	local iChannels
	
	-- The shortcut layer is either identity or 1x1 convolution
	local function shortcut(nInputPlane, nOutputPlane, stride)
		if nInputPlane ~= nOutputPlane then
			return nn.Sequential()
				:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
		else
			return nn.Identity()
		end
	end

	local function full_shortcut(nInputPlane, nOutputPlane, stride)
		if nInputPlane ~= nOutputPlane then
			return nn.Sequential()
				:add(pad(1,0,1,0))
				:add(nn.SpatialFullConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, 1, 1, 1, 1))
		else
			return nn.Identity()
		end
	end

	local function basic_block(n, stride)
		stride = stride or 1
		local nInputPlane = iChannels
		iChannels = n
  	-- Convolutions  
  	local conv_block = nn.Sequential()
  
  	conv_block:add(normalization(nInputPlane))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(pad(1, 1, 1, 1))
  	conv_block:add(nn.SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 0, 0))

  	conv_block:add(normalization(n))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(pad(1, 1, 1, 1))
  	conv_block:add(nn.SpatialConvolution(n, n, 3, 3, 1, 1, 0, 0))

  	local concat = nn.ConcatTable():add(conv_block):add(shortcut(nInputPlane, n, stride))
  
  	-- Sum
  	local res_block = nn.Sequential()
  	res_block:add(concat)
  	res_block:add(nn.CAddTable())
  	return res_block
	end

	local function bottleneck(n, stride)
		stride = stride or 1
		local nInputPlane = iChannels
		iChannels = 4 * n
		-- Convolutions  
  	local conv_block = nn.Sequential()
  
  	conv_block:add(normalization(nInputPlane))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(nn.SpatialConvolution(nInputPlane, n, 1, 1, 1, 1, 0, 0))

  	conv_block:add(normalization(n))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(pad(1, 1, 1, 1))
  	conv_block:add(nn.SpatialConvolution(n, n, 3, 3, stride, stride, 0, 0))

  	conv_block:add(normalization(n))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(nn.SpatialConvolution(n, n*4, 1, 1, 1, 1, 0, 0))

  	local concat = nn.ConcatTable():add(conv_block):add(shortcut(nInputPlane, n*4, stride))
  
  	-- Sum
  	local res_block = nn.Sequential()
  	res_block:add(concat)
  	res_block:add(nn.CAddTable())
  	return res_block
	end

	local function full_bottleneck(n, stride)
		stride = stride or 1
		local nInputPlane = iChannels
		iChannels = 4 * n
		-- Convolutions  
  	local conv_block = nn.Sequential()
  
  	conv_block:add(normalization(nInputPlane))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(nn.SpatialConvolution(nInputPlane, n, 1, 1, 1, 1, 0, 0))

  	conv_block:add(normalization(n))
  	conv_block:add(nn.ReLU(true))

		if stride~=1 then
  		conv_block:add(nn.SpatialFullConvolution(n, n, 3, 3, stride, stride, 1, 1, 1, 1))
		else
  		conv_block:add(pad(1, 1, 1, 1))
  		conv_block:add(nn.SpatialConvolution(n, n, 3, 3, 1, 1, 0, 0))
		end
  	conv_block:add(normalization(n))
  	conv_block:add(nn.ReLU(true))
  	conv_block:add(nn.SpatialConvolution(n, n*4, 1, 1, 1, 1, 0, 0))

  	local concat = nn.ConcatTable():add(conv_block):add(full_shortcut(nInputPlane, n*4, stride))
  
  	-- Sum
  	local res_block = nn.Sequential()
  	res_block:add(concat)
  	res_block:add(nn.CAddTable())
  	return res_block
	end
	
	local function layer(block, features, count, stride)
		local s = nn.Sequential()
		for i=1,count do
			s:add(block(features, i==1 and stride or 1))
		end
		return s
	end

	local model = nn.Sequential()
	model.cNetsNum = {}

	-- 512x512
	model:add(normalization(3))
	model:add(pad(3, 3, 3, 3))
	model:add(nn.SpatialConvolution(3, 64, 7, 7, 1, 1, 0, 0))
	model:add(normalization(64))
	model:add(nn.ReLU(true))

	iChannels = 64
	local block = bottleneck -- basic_block
	
	-- 512x512x64
	model:add(nn.Inspiration(iChannels))
	table.insert(model.cNetsNum,#model)
	model:add(normalization(iChannels))
	model:add(nn.ReLU(true))
	model:add(layer(block, 32, opt.model_nres, 2))

	-- 256x256x128
	model:add(nn.Inspiration(iChannels))
	table.insert(model.cNetsNum,#model)
	model:add(normalization(iChannels))
	model:add(nn.ReLU(true))
	model:add(layer(block, 64, opt.model_nres, 2))
	
	-- 128x128x256
	model:add(nn.Inspiration(iChannels))
	table.insert(model.cNetsNum,#model)
	model:add(normalization(iChannels))
	model:add(nn.ReLU(true))
	model:add(layer(block, 128, opt.model_nres, 2))
	
	-- 64x64x512
	model:add(nn.Inspiration(iChannels))
	table.insert(model.cNetsNum,#model)
	model:add(normalization(iChannels))
	model:add(nn.ReLU(true))
	
	block = full_bottleneck
	model:add(layer(block, 64, opt.model_nres, 2))
	model:add(layer(block, 32, opt.model_nres, 2))
	model:add(layer(block, 16, opt.model_nres, 2))

	model:add(normalization(64))
	model:add(nn.ReLU(true))

	model:add(pad(3, 3, 3, 3))
	model:add(nn.SpatialConvolution(64, 3, 7, 7, 1, 1, 0, 0))
	model:add(nn.Tanh())
	model:add(nn.MulConstant(opt.tanh_constant))
	model:add(nn.TotalVariation(opt.tv_strength))

	function model:setTarget(feat, dtype)
		for i=1,#model.cNetsNum do
			model.modules[model.cNetsNum[i]]:setTarget(feat[i]:type(dtype))
		end
	end
	return model
end

function M.createCNets(opt)
	-- The descriptive network in the paper
  local cnet = torch.load(opt.loss_network)
	cnet:evaluate()
	cnet.style_layers = {}

  -- Set up calibrate layers
  for i, layer_string in ipairs(opt.style_layers) do
		local calibrator = nn.Calibrate()
		layer_utils.insert_after(cnet, layer_string, calibrator)
		table.insert(cnet.style_layers, calibrator)
  end	
	layer_utils.trim_network(cnet)

	function cnet:calibrate(input)
		cnet:forward(input)
		local feat = {}
		for i, calibrator in ipairs(cnet.style_layers) do
				table.insert(feat, calibrator:getGram())
		end
		return feat
	end

	return cnet
end

return M
