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

local Calibrate, parent = torch.class('nn.Calibrate', 'nn.Module')

function Calibrate:__init()
	parent.__init(self)
	self.gram = torch.Tensor()
	self.calibrator = nn.GramMatrix()
end

function Calibrate:updateOutput(input)
	assert(self and input)
	self.gram = self.calibrator:forward(input):squeeze()
	return input
end

function Calibrate:updateGradInput(input, gradOutput)
	assert(self and gradOutput)
	self.gradInput = gradOutput
	return self.gradInput
end

function Calibrate:getGram()
	return self.gram:clone()
end
