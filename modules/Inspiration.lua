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

local Inspiration, parent = torch.class('nn.Inspiration', 'nn.Module')

local function isint(x) 
	return type(x) == 'number' and x == math.floor(x) 
end

function Inspiration:__init(C)
	parent.__init(self)
	assert(self and C, 'should specify C')
	assert(isint(C), 'C should be integers')

	self.C = C
	self.MM_WG = nn.MM()
	self.MM_PX = nn.MM(true, false)
	self.target = torch.Tensor(C,C)
	self.Weight = torch.Tensor(C,C)
	self.P = torch.Tensor(C,C)

	self.gradWeight = torch.Tensor(C, C)
	self.gradWG = {torch.Tensor(C, C), torch.Tensor(C, C)}
	self.gradPX = {torch.Tensor(), torch.Tensor()}
	self.gradInput = torch.Tensor()
	self:reset()
end

function Inspiration:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(2)
	else
		stdv = 1./math.sqrt(self.C)
	end
	self.Weight:uniform(-stdv, stdv)
	self.target:uniform(-stdv, stdv)
	return self
end

function Inspiration:setTarget(nT)
	assert(self and image)
	self.target = nT
end

function Inspiration:updateOutput(input)
	assert(self)
	-- P=WG Y=XP
	--self.output:resizeAs(input)
	if input:dim() == 3 then
		self.P = self.MM_WG:forward({self.Weight, self.target})
		self.output = self.MM_PX:forward({self.P, input:view(self.C,-1)}):viewAs(input)
	elseif input:dim() == 4 then
		local B = input:size(1)
		self.P = self.MM_WG:forward({self.Weight, self.target})
		self.output = self.MM_PX:forward({self.P:add_dummy():expand(B,self.C,self.C), input:view(B,self.C,-1)}):viewAs(input)
	else
		error('Unsupported dimention for Inspiration layer')
	end
	return self.output
end

function Inspiration:updateGradInput(input, gradOutput)
	assert(self and self.gradInput)	

	--self.gradInput:resizeAs(input):fill(0)
	if input:dim() == 3 then
		self.gradPX = self.MM_PX:backward({self.P, input:view(self.C,-1)}, gradOutput:view(self.C,-1))
	elseif input:dim() == 4 then
		local B = input:size(1)
		self.gradPX = self.MM_PX:backward({self.P:add_dummy():expand(B,self.C,self.C), input:view(B,self.C,-1)}, gradOutput:view(B,self.C,-1))
	else
		error('Unsupported dimention for Inspiration layer')
	end

	self.gradInput = self.gradPX[2]:viewAs(input)
	return self.gradInput
end

function Inspiration:accGradParameters(input, gradOutput, scale)
	assert(self)
	scale = scale or 1

	if input:dim() == 3 then
		self.gradWG = self.MM_WG:backward({self.Weight, self.target}, self.gradPX[1])
		self.gradWeight = scale * self.gradWG[1]
	elseif input:dim() == 4 then
		self.gradWG = self.MM_WG:backward({self.Weight, self.target}, self.gradPX[1]:sum(1):squeeze())
		self.gradWeight = scale * self.gradWG[1]
	else
		error('Unsupported dimention for Inspiration layer')
	end
end

function Inspiration:__tostring__()
	return torch.type(self) ..
		string.format(
			'(%dxHxW, -> %dxHxW)',
			self.C, self.C
			)
end
