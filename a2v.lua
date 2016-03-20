require 'torch';
require 'nn';
require 'cunn';
require 'optim';
require 'lmdb';
require 'os';
require 'xlua'
require 'rnn'
require 'os'

local db = lmdb.env{Path = 'fullDB', Name = 'fullDB'}
db:open()
local txn = db:txn(true)
author_size = txn:get('na')
print('na = '..author_size)
paper_size = txn:get('np')
print('np = '..paper_size)
hidden_size = 100
learning_rate = 0.1
decay = 0.001
batch_size = 512
max_epochs = 25
iscuda = true
hid2 = 50
pp2Seq = txn:get('pp2Seq')
seq2auth = txn:get('seq2auth')

paper_context_lookup = nn.LookupTable(paper_size, hidden_size)
print('Loading pre-trained paper embeddings...')
local f = io.open('doc2vec_embeddings.txt', 'r')
xlua.progress(1, 227949)
c = 0
pos = 0
for line in f:lines() do
	local result = string.gsub(line, '\n', '')
	local arr = string.split(result, '\t')
	local row = pp2Seq[arr[1]]
	c = c + 1
	if row ~= nil then
		local tensor = torch.Tensor(#arr - 1)
		for i = 2, #arr do
			tensor[i-1] = tonumber(arr[i])
		end
		paper_context_lookup.weight[row] = tensor
		pos = pos + 1
	end
    if c % 100 == 0 then
		xlua.progress(c, 227949)
	end
end
print('# init. '..pos)
f:close()
collectgarbage()
xlua.progress(227949, 227949)
print('Paper embeddings loaded')

print('Building model1...')
w_1 = nn.Linear(hidden_size, hid2)
author_lookup = nn.LookupTable(author_size, hidden_size)
paper_context_model = nn.Sequential()
paper_context_model:add(paper_context_lookup)
paper_context_model:add(w_1)
paper_context_model:add(nn.Tanh())
author_model = nn.Sequential()
author_model:add(author_lookup)
author_model:add(w_1:clone('weight', 'bias', 'gradWeight', 'gradBias'))
author_model:add(nn.Tanh())
combo11 = nn.Sequential()
combo11:add(nn.ParallelTable())
combo11.modules[1]:add(paper_context_model)
combo11.modules[1]:add(author_model)
combo11:add(nn.CMulTable())
combo11:add(nn.Linear(hid2, hid2))
paper_context_model_clone = paper_context_model:clone('weight', 'bias', 'gradWeight', 'gradBias')
author_model_clone = author_model:clone('weight', 'bias', 'gradWeight', 'gradBias')
combo12 = nn.Sequential()
combo12:add(nn.ParallelTable())
combo12.modules[1]:add(paper_context_model_clone)
combo12.modules[1]:add(author_model_clone)
combo12:add(nn.CSubTable())
combo12:add(nn.Abs())
combo12:add(nn.Linear(hid2, hid2))
model1 = nn.Sequential()
model1:add(nn.ConcatTable())
model1.modules[1]:add(combo11)
model1.modules[1]:add(combo12)
model1:add(nn.CAddTable())
model1:add(nn.Tanh())
model1:add(nn.Linear(hid2,2))
model1:add(nn.LogSoftMax())

print('Building model2...')
w_2 = nn.Linear(hidden_size, hid2)
author_model2_lookup = author_model:clone('weight', 'bias', 'gradWeight', 'gradBias')
author_context_lookup = nn.LookupTable(author_size, hidden_size)
author_context_model = nn.Sequential()
author_context_model:add(author_context_lookup)
author_context_model:add(w_2)
author_context_model:add(nn.Tanh())
author_model2 = nn.Sequential()
author_model2:add(author_model2_lookup)
author_model2:add(w_2:clone('weight', 'bias', 'gradWeight', 'gradBias'))
author_model2:add(nn.Tanh())
combo21 = nn.Sequential()
combo21:add(nn.ParallelTable())
combo21.modules[1]:add(author_context_model)
combo21.modules[1]:add(author_model2)
combo21:add(nn.CMulTable())
combo21:add(nn.Linear(hid2, hid2))
author_context_model_clone = author_context_model:clone('weight', 'bias', 'gradWeight', 'gradBias')
author_model_clone_2 = author_model2:clone('weight', 'bias', 'gradWeight', 'gradBias')
combo22 = nn.Sequential()
combo22:add(nn.ParallelTable())
combo22.modules[1]:add(author_context_model_clone)
combo22.modules[1]:add(author_model_clone_2)
combo22:add(nn.CSubTable())
combo22:add(nn.Abs())
combo22:add(nn.Linear(hid2, hid2))
model2 = nn.Sequential()
model2:add(nn.ConcatTable())
model2.modules[1]:add(combo21)
model2.modules[1]:add(combo22)
model2:add(nn.CAddTable())
model2:add(nn.Tanh())
model2:add(nn.Linear(hid2,2))
model2:add(nn.LogSoftMax())

model1 = nn.Sequencer(model1)
model2 = nn.Sequencer(model2)
--criterion1 = nn.CrossEntropyCriterion()
criterion1 = nn.ClassNLLCriterion()
criterion1 = nn.SequencerCriterion(criterion1)
--criterion2 = nn.CrossEntropyCriterion()
criterion2 = nn.ClassNLLCriterion()
criterion2 = nn.SequencerCriterion(criterion2)
if iscuda == true then
	model1 = model1:cuda()
	model2 = model2:cuda()
	criterion1 = criterion1:cuda()
	criterion2 = criterion2:cuda()
end

param_model1, grad_params_model1 = model1:getParameters()
param_model2, grad_params_model2 = model2:getParameters()
author_batch_input,author_batch_label, paper_batch_input, paper_batch_label  = {}, {}, {}, {}

feval1 = function(x)
	--[[
	param_model1:copy(x)
	grad_params_model1:zero()
	local loss = 0
	for i = 1, #paper_batch_input do
		pred1 = model1:forward(paper_batch_input[i])		
		local curLoss = criterion1:forward(pred1, paper_batch_label[i])
		if curLoss < 0 or loss < 0 then
			print(pred1)
			print(paper_batch_label[i])
			os.exit(0)
		end
		loss = loss + curLoss
		grads = criterion1:backward(pred1, paper_batch_label[i])
		model1:backward(paper_batch_input[i], grads)		
	end
	grad_params_model1:div(#paper_batch_input)
	return loss / #paper_batch_input, grad_params_model1
	]]--
	
	param_model1:copy(x)
	grad_params_model1:zero()
	local loss = 0
	pred1 = model1:forward(paper_batch_input)
	loss = loss + criterion1:forward(pred1, paper_batch_label)
	grads = criterion1:backward(pred1, paper_batch_label)
	model1:backward(paper_batch_input, grads)
	grad_params_model1:div(#paper_batch_input)
	return loss / #paper_batch_input, grad_params_model1
end
feval2 = function(x)
	--[[
	param_model2:copy(x)
	grad_params_model2:zero()
	local loss = 0
	for i = 1, #author_batch_input do
		pred1 = model2:forward(author_batch_input[i])		
		local curLoss = criterion2:forward(pred1, author_batch_label[i])
		if curLoss < 0 or loss < 0 then
			print(pred1)
			print(author_batch_label[i])
			os.exit(0)
		end
		loss = loss + curLoss
		grads = criterion2:backward(pred1, author_batch_label[i])
		model2:backward(author_batch_input[i], grads)		
	end
	grad_params_model2:div(#author_batch_input)
	return loss / #author_batch_input, grad_params_model2
	]]--

	param_model2:copy(x)
	grad_params_model2:zero()
	local loss = 0
	pred1 = model2:forward(author_batch_input)
	loss = loss + criterion2:forward(pred1, author_batch_label)
	grads = criterion2:backward(pred1, author_batch_label)
	model2:backward(author_batch_input, grads)
	grad_params_model2:div(#author_batch_input)
	return loss / #author_batch_input, grad_params_model2
end
optim_state = {learningRate = learning_rate, alpha = decay}
function save(file)
	print('saving...')
	local start = sys.clock()
	local f = io.open(file, 'w')
	local row, col = author_size, hidden_size
	for i = 1, row do
		res = seq2auth[i]
		for j = 1, col do
			res = res .. '\t' .. author_lookup.weight[i][j]
		end 
		f:write(res .. '\n')
	end
	f.close()
	print(string.format("Done in %.2f minutes.\n", (sys.clock() - start)/60))
end

local db = lmdb.env{Path = 'fullDB', Name = 'fullDB'}
db:open()
local txn = db:txn(true)
paper_vector_size = tonumber(txn:get('paper_size'))
author_vector_size = tonumber(txn:get('author_size'))
for i = 1, max_epochs do
	local epoch_start = sys.clock()
	print(string.format("epoch: %d", i))
	batch_count = 0
	epoch_paper_loss = 0
	start_time = os.time()	
	local parr = torch.randperm(paper_vector_size)
	paper_batch = {}
	xlua.progress(1, paper_vector_size)

	local model1_start = sys.clock()
	for j = 1, paper_vector_size do
		local input = txn:get('paper_' .. tostring(parr[j]))
		paper_batch_input = input[1]
		paper_batch_label = input[2]
		--table.insert(paper_batch_input, input[1])
		--table.insert(paper_batch_label, input[2])
		_, loss = optim.sgd(feval1, param_model1, optim_state)
		epoch_paper_loss = epoch_paper_loss + loss[1]
		batch_count = batch_count + 1
		paper_batch_input, paper_batch_label = nil, nil
		paper_batch_input, paper_batch_label = {}, {}
		--[[
		--table.insert(paper_batch_input, {torch.Tensor{1},torch.Tensor{2}})
		--table.insert(paper_batch_label, torch.IntTensor{1})

		if #paper_batch_input == batch_size then
			_, loss = optim.sgd(feval1, param_model1, optim_state)
			epoch_paper_loss = epoch_paper_loss + loss[1]
			batch_count = batch_count + 1
			paper_batch_input, paper_batch_label = nil, nil
			paper_batch_input, paper_batch_label = {}, {}
			--collectgarbage()
		end
		]]--
		--if batch_count % 2 == 0 then collectgarbage() end
		if j % 2 == 0 then 
			collectgarbage()
			xlua.progress(j, paper_vector_size) 
		end
	end
	--[[
	if #paper_batch_input ~= 0 then
		_, loss = optim.sgd(feval1, param_model1, optim_state)
		epoch_paper_loss = epoch_paper_loss + loss[1]
		batch_count = batch_count + 1
		paper_batch_input, paper_batch_label = nil, nil
		paper_batch_input, paper_batch_label = {}, {}
		--collectgarbage()
	end
	]]--

	collectgarbage()
	xlua.progress(paper_vector_size, paper_vector_size)
	print(string.format("paper_loss: %f, time: %.2f min", (epoch_paper_loss / batch_count), ((sys.clock()-model1_start)/60)))

	local model2_start = sys.clock()
	local aarr = torch.randperm(author_vector_size)
	author_batch = {}
	epoch_author_loss = 0
	batch_count = 0
	xlua.progress(1, author_vector_size)

	for j = 1, author_vector_size do
		local input = txn:get('author_' .. tostring(aarr[j]))
		author_batch_input = input[1]
		author_batch_label = input[2]
		--table.insert(author_batch_input, input[1])
		--table.insert(author_batch_label, input[2])
		--table.insert(author_batch_input, {torch.Tensor{1},torch.Tensor{2}})
		--table.insert(author_batch_label, torch.IntTensor{1})
		--if #author_batch_input == batch_size then
			_, loss = optim.sgd(feval2, param_model2, optim_state)
			epoch_author_loss = epoch_author_loss + loss[1]
			batch_count = batch_count + 1
			author_batch_input, author_batch_label = nil, nil		
			author_batch_input, author_batch_label = {}, {}
		--end
		--if batch_count % 2 == 0 then collectgarbage() end
		if j % 2 == 0 then 
			xlua.progress(j, author_vector_size) 
			collectgarbage()
		end
	end
	--[[
	if #author_batch_label ~= 0 then
		_, loss = optim.sgd(feval2, param_model2, optim_state)
		epoch_author_loss = epoch_author_loss + loss[1]
		batch_count = batch_count + 1
		author_batch_input, author_batch_label = nil, nil		
		author_batch_input, author_batch_label = {}, {}
	end
	]]--

	xlua.progress(author_vector_size, author_vector_size)
	print(string.format("author_loss: %f, time: %.2f min", (epoch_author_loss / batch_count), ((sys.clock()-model2_start)/60)))

	print('Saving model...')
	save('a2v_m12_' .. tostring(i) .. '.txt') 
	if i ~= 1 then
		os.execute('rm a2v_m12_'..(i-1)..'.txt')
	end
	collectgarbage()
	print(string.format("epoch %d done in %.2f min", i, ((sys.clock()-epoch_start)/60)))
end
txn:abort()
db:close()
