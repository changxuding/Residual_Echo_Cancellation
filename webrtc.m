function [st, webrtc_out] = webrtc(st, laec_out, mic_frame, spk_frame)

    N = st.frame_size;
    N1 = st.half_bin;
    gamma = 0.93;

    % find the max weight index
    wEn = sum(real(st.coef_adf.*conj(st.coef_adf)));
    [~, idx] = max(wEn);

    st.ee = [st.ee(N1:end); laec_out];
    st.dd = [st.dd(N1:end); mic_frame];
    st.xx = [st.xx(N1:end); spk_frame];

    tmp = fft(st.ee .* st.window);
    EE = tmp(1:N1);
    tmp = fft(st.dd .* st.window);
    DD = tmp(1:N1);      
    tmp = fft(st.xx .* st.window);
    XX = tmp(1:N1);
    st.XX = [XX ,st.XX(:, 1:end-1)];
    XX = st.XX(:,idx);

    st.Se = gamma*st.Se + (1-gamma)*real(EE.*conj(EE));
    st.Sd = gamma*st.Sd + (1-gamma)*real(DD.*conj(DD));
    st.Sx = gamma*st.Sx + (1-gamma)*real(XX.*conj(XX));

    % coherence
    st.Sxd = gamma*st.Sxd + (1-gamma)*(XX.*conj(XX));
    st.Sed = gamma*st.Sed + (1-gamma)*(EE.*conj(EE));

    cohed = real(st.Sed.*conj(st.Sed))./(st.Se.*st.Sd + 1e-10);
    cohxd = real(st.Sxd.*conj(st.Sxd))./(st.Sx.*st.Sd + 1e-10);
    % freqSm = 0.55;
    % cohxd(2:end) = filter(freqSm, [1 -(1-freqSm)], cohxd(2:end));
    % cohxd(end:2) = filter(freqSm, [1 -(1-freqSm)], cohxd(end:2));
    hnled = min(1 - cohxd, cohed);
    hnled = max(0.05, hnled);
    EE = EE.*(hnled);

    % OLA
    tmp = [EE; conj(EE(end-1 :-1:2))];
    e_buf = st.window.*real(ifft(tmp));
    webrtc_out = st.webrtc_buf(end-N+1:end) + e_buf(1:N);
    st.webrtc_buf = e_buf;
end