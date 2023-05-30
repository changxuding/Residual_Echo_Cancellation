function [st, speex_out] = speex(st, laec_out)
    N = st.frame_size;
    N1 = st.half_bin;
    Pey = 1;
    Pyy = 1;
    Syy = 0;
    See = 0;
    st.ee = [zeros(N,1); laec_out];
    st.et = [st.et(N1:end); laec_out];
    EE = fft(st.ee);
    YY = fft(st.yy);
    resPsd = abs(EE(1:N1)).^2;
    estEchoPsd = abs(YY(1:N1)).^2;
    Eh_cur = resPsd - st.Eh;
    Yh_cur = estEchoPsd - st.Yh;
    Pey = Pey + sum(Eh_cur.*Yh_cur) ;
    Pyy = Pyy + sum(Yh_cur.^2);
    st.Eh = (1-st.spec_average)*st.Eh + st.spec_average*resPsd;
    st.Yh = (1-st.spec_average)*st.Yh + st.spec_average*estEchoPsd;
    Pyy = sqrt(Pyy);
    Pey = Pey/Pyy;
    
    tmp32 = min(st.beta0*Syy, st.beta_max*See);
    alpha = tmp32/ See;
    alpha_1 = 1- alpha;
    
    % Update correlations
    st.Pey = alpha_1*st.Pey + alpha*Pey;
    st.Pyy = alpha_1*st.Pyy + alpha*Pyy;
    st.Pyy = max(st.Pyy, 1);
    st.Pey = max(st.Pey, st.MIN_LEAK * st.Pyy);
    st.Pey = min(st.Pey, st.Pyy);

    % leak_estimate
    leak_estimate = st.Pey/st.Pyy;
    if leak_estimate > 0.5
        leak_estimate = 1;
    else
        leak_estimate = leak_estimate * 2;
    end

    % residual suppression use stsa
    EE = fft(st.et .*st.window);
    resPsd = abs(EE(1:N1)).^2;
    YY = fft(st.yt .*st.window);
    estEchoPsd = abs(YY(1:N1)).^2;

    residual_echo = estEchoPsd * leak_estimate * 20;
    if st.update_cnt==0
        st.echo_noise = residual_echo;
    else
        st.echo_noise = max(0.85*st.echo_noise, residual_echo);
    end
    postSnr = resPsd ./ st.echo_noise;
    postSnr = min(postSnr, 100);
    if st.update_cnt==0
        st.old_ps = resPsd;
    end
    alpha = 0.2 + 0.8 * st.old_ps ./ (st.old_ps + st.echo_noise);
    priSnr = alpha .* max(0, postSnr - 1) + (1 - alpha) .* st.old_ps./st.echo_noise;
    priSnr = min(priSnr, 100);
    wiener_gain = priSnr ./ (priSnr + 1);
    if(0)
        v = postSnr .* wiener_gain;
        for i=1:N1
            if v(i)<1
                st.gain(i) = gamma(1.5)*sqrt(v(i))/postSnr(i)*exp(-v(i)/2).*(1+v(i))...
                    .*besseli(0,v(i)/2) + v(i) .* besseli(1,v(i)/2);
            else
                st.gain(i) = wiener_gain(i);
            end
        end
    else
        st.gain = wiener_gain;
    end
    st.gain = min(1, st.gain);
    st.gain = max(0.05, st.gain);
    st.old_ps = 0.8*st.old_ps + 0.2*st.gain.*resPsd;
    tmp = st.gain.*EE(1:N1);
    fft_in = [tmp; conj(tmp(end-1:-1:2))];
    e_buf = st.window.*real(ifft(fft_in));
    speex_out = st.speex_buf(end-N+1:end) + e_buf(1:N);
    st.speex_buf = e_buf;
end