input_domain = stk_hrect ([                                  ...
    0.05    100   63070    990   63.1    700  1120   9855;   ...
    0.15  50000  115600   1110  116.0    820  1680  12045],  ...
    {'rw',  'r',    'Tu',  'Hu',  'Tl',  'Hl',  'L',  'Kw'});

dim = size (input_domain, 2);

sigma = 10;

n_obs = 5e4;
n_tst = 1e4;

x_obs = stk_sampling_randunif (n_obs, dim, input_domain);
y_obs = stk_testfun_borehole (x_obs) + sigma * randn (n_obs, 1);

x_tst = stk_sampling_randunif (n_tst, dim, input_domain);
y_tst = stk_testfun_borehole (x_tst) + sigma * randn (n_tst, 1);


%% Quick test, using only 200 observations

n = 200;

M_prior = stk_model (@stk_materncov_aniso, dim);
M_prior.lognoisevariance = nan;

M_post = stk_model_gpposterior (M_prior, x_obs(1:n,:), y_obs(1:n,:));

y_prd = stk_predict (M_post, x_tst);

figure (1);  clf ();
plot (y_prd.mean, y_tst, 'k.');  hold on;
plot (xlim, xlim, 'r--');
stk_labels ('emulator mean', 'test data', 'FontSize', 12);


%% Save

data_obs = double ([x_obs, y_obs]);
data_tst = double ([x_tst, y_tst]);

writematrix (data_obs, 'borehole_obs.csv');
writematrix (data_tst, 'borehole_tst.csv');
