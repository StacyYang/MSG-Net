-- adding first dummy dimension (by Dmitry Ulyanov)

function torch.FloatTensor:add_dummy()
  local sz = self:size()
  local new_sz = torch.Tensor(sz:size()+1)
  new_sz[1] = 1
  new_sz:narrow(1,2,sz:size()):copy(torch.Tensor{sz:totable()})

  if self:isContiguous() then
    return self:view(new_sz:long():storage())
  else
    return self:reshape(new_sz:long():storage())
  end
end

torch.Tensor.add_dummy = torch.FloatTensor.add_dummy
if cutorch then 
  torch.CudaTensor.add_dummy = torch.FloatTensor.add_dummy
end
