%% LDPC Processing for DL-SCH and UL-SCH
%
% This example highlights the low-density parity-check (LDPC) coding chain
% for the 5G NR downlink and uplink shared transport channels (DL-SCH and 
% UL-SCH).

% Copyright 2018-2019 The MathWorks, Inc.

% https://www.mathworks.com/help/5g/gs/ldpc-processing-chain-for-dl-sch.html
    
% Example was slightly modified 
clear all;

%% Shared Channel Parameters
%
% The example uses the DL-SCH to describe the processing, which also
% applies to the UL-SCH.
%
% Select parameters for a transport block transmitted on the downlink
% shared (DL-SCH) channel.

% s = rng(100);              % Set RNG state for repeatability

% R_list = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679, 340, 378, 434, ...
%           490, 553, 616, 658, 438, 466, 517, 567, 616, 666, 719, 772, 772, ...
%           822, 873, 910, 948];
% M_list = [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   4,   4,   4, ...
%           4,   4,   4,   4,   6,   6,   6,   6,   6,   6,   6,   6,   6, ...  
%           6,   6,   6,   6];

R_list = [567];
M_list = [6];

% seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
N_SEEDS = 1000;
seed_list = linspace(1, N_SEEDS, N_SEEDS);

A = [1024, 3072, 4096];              % Transport block length, positive integer
rv = 0;                % Redundancy version, 0-3
nlayers = 1;           % Number of layers, 1-4 for a transport block
EbNo = -10:0.1:25;     % EbNo in dB

%%
% Based on the selected transport block length and target coding rate,
% DL-SCH coding parameters are determined using the
% <docid:5g_ref#mw_function_nrDLSCHInfo nrDLSCHInfo> function.

figure;

for n_mcs = 1:length(R_list)
    
    disp(['computing MCS ', num2str(n_mcs - 1)]);
    rate = R_list(n_mcs)/1024;       % Target code rate, 0<R<1
    
    if M_list(n_mcs) == 2
        modulation = 'QPSK';   % Modulation scheme, QPSK, 16QAM, 64QAM, 256QAM
    elseif M_list(n_mcs) == 4
        modulation = '16QAM';
    elseif M_list(n_mcs) == 6
        modulation = '64QAM';
    end
    
    for n_seed = 1:length(seed_list)
        s = seed_list(n_seed);
        for k = 1:length(EbNo)
            % DL-SCH coding parameters
            cbsInfo = nrDLSCHInfo(A(n_mcs),rate);
            %disp('DL-SCH coding parameters')
            %disp(cbsInfo)

            bps = 2;
            EsNo = EbNo(k) + 10*log10(bps);       
            snrdB = EsNo + 10*log10(rate);       % in dB
            noiseVar = 1./(10.^(snrdB/10)); 

            % Random data generation
            in = randi([0 1],A(n_mcs),1,'int8');

            % Transport block CRC attachment 
            tbIn = nrCRCEncode(in,cbsInfo.CRC);

            % Code block segmentation and CRC attachment
            cbsIn = nrCodeBlockSegmentLDPC(tbIn,cbsInfo.BGN);

            % LDPC encoding
            enc = nrLDPCEncode(cbsIn,cbsInfo.BGN);

            % Rate matching and code block concatenation
            outlen = ceil(A(n_mcs)/rate);
            chIn = nrRateMatchLDPC(enc,outlen,rv,modulation,nlayers);
            chIn = nrSymbolModulate(chIn,modulation);

            %% Channel
            %
            % A simple bipolar channel with no noise is used for this example.
            % With the full PDSCH or PUSCH processing, one can consider fading
            % channels, AWGN and other RF impairments as well.

            % chOut = double(1-2*(chIn));
            awgn_chan = comm.AWGNChannel('NoiseMethod','Variance','Variance',noiseVar);
%             awgn_chan = comm.RayleighChannel('SampleRate',3.84e9, ...
%             'PathDelays',[0 1000 12000]*1e-9, ...
%             'AveragePathGains',[0 -5.9 -8.9], ...
%             'MaximumDopplerShift',0);
            chOut = awgn_chan(double(chIn));

             % Soft demodulate
            rxLLR = nrSymbolDemodulate(chOut,modulation,noiseVar);

            %% Receive Processing using LDPC Decoding
            %
            % The receive end processing for the DL-SCH channel comprises of
            % the corresponding dual operations to the transmit end that include
            %
            % * Rate recovery 
            % * LDPC decoding
            % * Code block desegmentation and CRC decoding
            % * Transport block CRC decoding
            %
            % Each of these stages is performed by a function as shown next.

            % Rate recovery
            raterec = nrRateRecoverLDPC(rxLLR,A(n_mcs),rate,rv,modulation,nlayers);

            % LDPC decoding
            decBits = nrLDPCDecode(raterec,cbsInfo.BGN,25);

            % Code block desegmentation and CRC decoding
            [blk,blkErr] = nrCodeBlockDesegmentLDPC(decBits,cbsInfo.BGN,A(n_mcs)+cbsInfo.L);

            %disp(['CRC error per code-block: [' num2str(blkErr) ']'])

            % Transport block CRC decoding
            [out,tbErr] = nrCRCDecode(blk,cbsInfo.CRC);

            % disp(['CRC error: ' num2str(tbErr)])

            % errStats = biterr(double(out), double(in))/A;
            % BER = errStats(1);
            check = isequal(double(out), double(in));
            BLER(n_seed, k) = ~check;
            SNR(n_seed, k) = snrdB;

            rng(s);
        end
    end

    BLER = sum(BLER)./N_SEEDS;
    SNR = mean(SNR);
    plot(SNR, BLER);
    hold on;
    
end

xlabel('SNR');
ylabel('BLER');
set(gca, 'YScale', 'log')
% legend({'MCS 0'; 'MCS 1'; 'MCS 2'; 'MCS 3'; 'MCS 4'; 'MCS 5'; 'MCS 6'; ...
%         'MCS 7'; 'MCS 8'; 'MCS 9'; 'MCS 10'; 'MCS 11'; 'MCS 12'; ...
%         'MCS 13'; 'MCS 14'; 'MCS 15'; 'MCS 16'; 'MCS 17'; 'MCS 18'; ...
%         'MCS 19'; 'MCS 20'; 'MCS 21'; 'MCS 22'; 'MCS 23'; 'MCS 24'; ...
%         'MCS 25'; 'MCS 26'; 'MCS 27'; 'MCS 28'});

legend({'MCS 20'});

grid on;
