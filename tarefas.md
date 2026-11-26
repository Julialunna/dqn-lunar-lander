For episode = 1, M… e inicialização do estado	train	Inicia cada episódio com env.reset().
For t = 1, T…	train	Repete a interação até o fim do episódio.
With probability ε select a random action… otherwise… argmax…	choose_action	Escolhe uma ação com ε-greedy usando a rede principal.
Execute action… observe reward…	train	Executa env.step(action).
Set s… and preprocess…	train	Obtém o próximo estado. No Lunar Lander, você já recebe um vetor de observações.
Store transition… in D	train	Armazena a experiência no buffer, incluindo o sinal de término.
Sample random minibatch…	learn	Sorteia experiências do buffer.
Set yⱼ…	learn	Calcula os alvos usando recompensas e a rede alvo.
Perform a gradient descent step…	learn	Calcula a perda e atualiza a rede principal.
Every C steps reset…	train	Copia periodicamente os pesos da principal para a alvo.