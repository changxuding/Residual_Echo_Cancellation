function [st, out_frame] = mdf_kalman(st, mic_frame, spk_frame)

    N = st.frame_size;
    N1 = st.half_bin;

    [mic_frame_dc, st.notch_mem] = filter_dc_notch16(mic_frame, st.notch_radius, N, st.notch_mem);

    st.x = [st.x(N1:end); spk_frame];
    X = fft(st.x);
    st.X = [X(1:N1), st.X(:,1:end-1)];
    X_power = abs(st.X).^2;

    %filter out
    st.Y = sum(st.X .* st.coef_adf, 2);
    Y_fft_in = [st.Y; conj(st.Y(end-1 :-1:2))];
    st.err_adf = ifft(Y_fft_in);
    st.err_adf(1:N) = mic_frame_dc - st.err_adf(N+1:end);
    if st.mode==2
        st.yy = [zeros(N, 1) ;st.err_adf(N1:end)];
        st.yt = [st.yt(N1:end) ;st.err_adf(N1:end)];
    end

    %kalman update
    st.err_adf = [zeros(N,1);st.err_adf(1:N)];
    err_fft = fft(st.err_adf); 
    Psi_s = abs(err_fft(1:N1)).^2;
    st.Psi_s = 0.9*st.Psi_s+0.1*Psi_s;
    Psi_e = sum(st.P .* X_power, 2)+ st.Psi_s;                                       
    Psi_e = repmat(Psi_e, 1, st.M);

    mu = 0.5*st.P./(Psi_e+ 1e-10);                                             
    H = mu .* X_power;
    H = max(H, 1e-4);
    H = min(H, 1);
    st.P = st.A2.*(1-0.5*H).*st.P + (1-st.A2)*(abs(st.coef_adf).^2);
    st.P = min(st.P,st.P_MAX);
    st.P = max(st.P,st.P_MIN);

    PHI = mu.*conj(st.X).*repmat(err_fft(1:N1),1,st.M);
    st.coef_adf = st.coef_adf + PHI;
    
    % aumdf
    if(0)
        update_idx = mod(st.update_cnt,st.M)+1;
        Y = st.coef_adf(:,update_idx);
        fft_in = [Y; conj(Y(end-1 :-1:2))];
        wtmp = ifft(fft_in);
        wtmp(N+1:end) = 0;
        res = fft(wtmp, st.win_size);
        st.coef_adf(:,update_idx) = res(1:N+1);
    %mdf
    else
        for update_idx = 1:st.M
            Y = st.coef_adf(:,update_idx);
            fft_in = [Y; conj(Y(end-1 :-1:2))];
            wtmp = ifft(fft_in);
            wtmp(N+1:end) = 0;
            res = fft(wtmp, st.win_size);
            st.coef_adf(:,update_idx) = res(1:N+1);
        end
    end
    st.update_cnt = st.update_cnt + 1;


    H_res = 1 - sum(mu .* st.X .* conj(st.X),2);
    Ek_Res = err_fft(1:N1).*H_res;
    fft_in_res = [Ek_Res; conj(Ek_Res(end-1 :-1:2))];
    en_t = ifft(fft_in_res);
    out_frame = en_t(N+1:end);

    function [out,mem] = filter_dc_notch16(in, radius, len, mem)
        out = zeros(size(in));
        den2 = radius*radius + .7*(1-radius)*(1-radius);
        for ii=1:len
            vin = in(ii);
            vout = mem(1) + vin;
            mem(1) = mem(2) + 2*(-vin + radius*vout);
            mem(2) = vin - (den2*vout);
            out(ii) = radius*vout; 
        end
    end
end

