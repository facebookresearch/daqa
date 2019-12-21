% Copyright (c) Facebook, Inc. and its affiliates.
% All rights reserved.
%
% This source code is licensed under the license found in the
% LICENSE file in the root directory of this source tree.
%

%% Setup Path
% Loudness Toolbox
% http://genesis-acoustics.com/en/loudness_online-32.html
path_to_loudness_toolbox = 'LoudnessToolbox 1.2';
path_to_wavs = 'events/*.wav';

addpath(genpath(path_to_loudness_toolbox))

%% Main
audio_files = dir(path_to_wavs);
cell_structs = cell(size(audio_files, 1), 1);
parfor i = 1:size(audio_files, 1)
   [audio, Fs] = audioread([audio_files(i).folder, '/', audio_files(i).name]);
   audio_rs = resample(audio, 48000, Fs);
   res = Loudness_TimeVaryingSound_Moore(audio_rs, 48000);
   % output.(audio_files(i).name(1:end-4)) = res.LTLmax; % remove '.wav'
   s = struct;
   s.name = (audio_files(i).name(1:end-4));
   s.loud = res.LTLmax;
   cell_structs{i} = s;
   disp('Done')
end

for i = 1:size(audio_files, 1)
   output.(cell_structs{i}.name) = cell_structs{i}.loud;
end

%% Save
jsonStr = jsonencode(output);
fid = fopen('daqa_loudness.json', 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);
