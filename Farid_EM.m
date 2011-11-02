%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Expectation/Maximization Algorithm implementation by Hany Farid     %
%     Comentários em português por Giuliano pro Oscar e pro André ;)      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;  % limpa os itens do espaço de trabalho, libera memória

%%% MAKE SIGNAL
N = 128; % seta um sinal de 128 samples
f = rand(1,N); % essa rand retorna uma matriz 1xN (vetor linha de N colunas) de rands (0;1)
f = f - min(f); % esses passos servem para
f = f / max(f); % fazer stretching do sinal para o intervalo [0;1]
g = f; %  e atribui-lo a g ;)

%%% ADD CORRELATION
for k = 2 : 4 : N % um for de 2 a N com saltos de 4 -> for (k = 2; k <= N; k += 4)
    g(k) = 0.5*f(k-1) + 0.5*f(k+1); % isso aqui serve pra fazer com que, a cada 4 samples haja relações entre seus vizinhos
end

%%% EM
alpha = rand(2,1); % INITIAL COEFFICIENTS
sigma = 0.005; % VARIANCE ON GAUSSIAN
delta = 10; % UNIFORM

while (1)
    %%% E-STEP
    for k = 2 : N-1
        r(k) = g(k) - (alpha(1) * g(k-1) + alpha(2) * g(k+1)); % RESIDUAL
        w(k) = exp(-r(k)^2 / sigma) / (exp(-r(k)^2 / sigma) + 1 / delta); % PROBABILITY
    end

    %%% PLOT
    subplot(211); % NAO FUCKING SEI O QUE EH ISSO
    stem(w); % isso plota w, a probabilidade
    set(gca, 'Xtick', 2:4:N, 'Ytick', [0 0.5 1]); % seta o comportamento dos eixos do gráfico de cima, as labels
    title(sprintf('[%.2f %.2f]', alpha(1), alpha(2))); % seta o título do gráfico de cima para mostrar os parâmetros
    axis([0 N 0 1.1]); % seta os eixos do gráfico de probabilidade
    grid on; % liga o grid no gráfico
    subplot(212); % NAO FUCKING SEI O QUE EH ISSO
    plot(fftshift(abs(fft(w)))); % esse é o gráfico de baixo, isso plota o shift do espectro da transformada de Fourier de w
    axis([0 N 0 50]); % seta os eixos do espectro pra 0 a N em x e 0 a 50 em y
    drawnow; pause(0.25); % manda desenhar os plots e pausa um tempo entre os cálculos

    %%% M-STEP
    M = [g(1:N-2)' g(3:N)'];
    b = g(2:N-1)';
    r = r(2:end); % remove edge point;
    w = w(2:end); % remove edge point
    W = diag(w); % constrói uma matrix W pondo os elementos do vetor w como elementos de sua diagonal e zerando o resto
    alpha_new = inv(M'*W*M)*M'*W*b; % WLS
    
    if (norm(alpha - alpha_new) < 0.01) % define o critério de parada do while
        break; % STOPPING CONDITION
    end
    
    alpha = alpha_new; % atualiza alpha

    %%% UPDATE SIGMA
    sigma = sum(w.*(r.^2)) /sum(w); % atualiza sigma
end
