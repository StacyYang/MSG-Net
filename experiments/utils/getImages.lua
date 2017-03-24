--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2017
--
-- Free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--	  conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--		derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

local sys = require 'sys'
local ffi = require 'ffi'
require 'paths'
require 'image'

local M={}
local Dataset = torch.class('texture.Dataset', M)

function Dataset:_findImages(dir)
	local imagePath = torch.CharTensor()

	----------------------------------------------------------------------
	-- Options for the GNU and BSD find command
	local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
	local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
	for i=2,#extensionList do
		findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
	end

	-- Find all the images using the find command
	local f = io.popen('find -L ' .. dir .. findOptions)

	local maxLength = -1
	local imagePaths = {}

	-- Generate a list of all the images 
	while true do
		local line = f:read('*line')
		if not line then break end

		local filename = paths.basename(line)
		local path = dir .. '/' .. filename

		table.insert(imagePaths, path)

		maxLength = math.max(maxLength, #path + 1)
	end

	f:close()

	-- Convert the generated list to a tensor for faster loading
	local nImages = #imagePaths
	local imagePath = torch.CharTensor(nImages, maxLength):zero()
	for i, path in ipairs(imagePaths) do
		ffi.copy(imagePath[i]:data(), path)
	end

	return imagePath
end

function Dataset:__init(imgDir)
	assert(self)
	assert(paths.dirp(imgDir), 'image directory not found: ' .. imgDir)
	self.imagePath = {self:_findImages(imgDir)}
end

function Dataset:_loadImage(path)
	local ok, input = pcall(function()
		return image.load(path, 3, 'float')
	end)

	-- Sometimes image.load fails because the file extension does not match the
	-- image format. In that case, use image.decompress on a ByteTensor.
	if not ok then
		local f = io.open(path, 'r')
		assert(f, 'Error reading: ' .. tostring(path))
		local data = f:read('*a')
		f:close()

		local b = torch.ByteTensor(string.len(data))
		ffi.copy(b:data(), data, b:size(1))

		input = image.decompress(b, 3, 'float')
	end

	return input
end

function Dataset:get(i)
	assert(self and self.imagePath)
	i = (i-1) % self:size() + 1
	local path = ffi.string(self.imagePath[1][i]:data())
	local image = self:_loadImage(path)
	return image
end

function Dataset:size()
	assert(self and self.imagePath)
	return self.imagePath[1]:size(1)
end

return M.Dataset
