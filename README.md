# **Fast and Efficient Creation of an Universal Quantum Gate Set Using Reinforcement Learning Methods**
***
![first_page_logo(1)](https://github.com/user-attachments/assets/c4db67bb-1a35-4607-bea8-564be516aa89)

**Author:** Pablo Darío Conte

M.Sc. Degree Candidate in Quantum Computing and Quantum Technologies

**Faculty Advisor:** Emmanuel Paspalakis

**Members of the Examination Committee:** Ioannis Thanopulos and Dionisis Stefanatos

Department of Electrical and Computer Engineering - Democritus University of Thrace (DECE)

Institute of Nanoscience and Nanotechnology - National Center of Scientific Research “Demokritos” (INN-D)

## **Abstract**
---

The design and implementation of universal quantum gate sets are foundational to the advancement of quantum computing,
enabling the realization of complex quantum algorithms and error correction protocols.
This thesis explores the use of reinforcement learning (RL)
methods to efficiently construct a universal quantum gate set comprising the Hadamard (H),
the π-8 (T), and controlled-NOT (CNOT) gates. These gates, recognized for their minimality
and universality, serve as essential building blocks for arbitrary quantum operations.


The problem is formulated as a control optimization task, where various RL agents are
deployed to determine the optimal Rabi Frequency, Detuning and coupling strength timing
of quantum pulses to implement these gates with high fidelity. The investigation includes
an evaluation of multiple reinforcement learning algorithms to assess their performance in
balancing computational efficiency, physical constraints, and scalability across single and
multi-qubit systems.

![H_Fid](https://github.com/user-attachments/assets/6bb2e9ec-b2df-42ab-ab12-34a60bb975b6)

The proposed approach is validated through numerical simulations, demonstrating the
ability of RL techniques to automate and enhance the design of universal gate sets. This
work contributes to the growing synergy between machine learning and quantum technologies, offering a flexible and scalable framework for optimizing quantum control.
The findings highlight the potential of RL-based methodologies in advancing practical and robust imple-
mentations of quantum computing.

**Keywords:** *Reinforcement Learning, Machine Learning, Deep Learning, Quantum Control, Quantum Computing, Quantum Technologies*

## **Acknowledgements**
---

First, I would like to express my deepest gratitude to my advisor, Prof. Emmanuel Paspalakis, for his invaluable guidance, encouragement, and unwavering support throughout
this journey. His expertise and insights have been instrumental in shaping this thesis and,
undoubtedly, my understanding of quantum technologies.


I am also deeply thankful to the members of my examination committee, Prof. Ioannis
Thanopulos and Prof. Dionisis Stefanatos, for their valuable feedback and constructive
discussions, which have significantly improved the quality of this work.


A special thanks goes to professors, colleagues and peers at the Democritus University
of Thrace (DUTh) and the Institute of Nanoscience and Nanotechnology (INN-
D) for their lectures, collaboration, and shared passion for research. Their camaraderie and
support have made this experience truly enriching.

To my partner of heart, Jessica, I extend my deepest thanks for their unwavering belief in
me, their constant encouragement, and their understanding during this challenging journey.
Her support has been my source of strength and motivation throughout this endeavor.

Furthermore, I wish to express my sincere gratitude to the Colegio de Matemáticas
Bourbaki, under the distinguished leadership of Prof. Carlos Alfonso Ruiz Guido, for their
generous support which partially funded my studies at this institution. Their commitment
to fostering advanced research and education has been instrumental in my academic journey
and I am deeply thankful for the opportunities they have provided.

Finally, I am grateful to the wider scientific community and pioneers of quantum
science whose visionary work has inspired and guided this research.

## Results
---

**H-Gate**

![H_Gate_Max_Fidelity_Trajectory_Log_Infidelity](https://github.com/user-attachments/assets/6e1635db-08b4-4855-8b59-50446c11234a)

![H_Gate_Log_Infidelity_Comparison](https://github.com/user-attachments/assets/5840d7fe-0095-4bd1-b219-b9f2f2a83d46)

**T-Gate**

![T_Gate_Max_Fidelity_Trajectory_Log_Infidelity](https://github.com/user-attachments/assets/79895aa2-9cc6-407f-bab9-5c303d54c89e)

![T_Gate_Log_Infidelity_Comparison](https://github.com/user-attachments/assets/a713b86d-6912-4fc7-9429-aaf56a143cf2)


**CNOT-Gate**

![CNOT_Gate_Max_Fidelity_Trajectory_Log_Infidelity](https://github.com/user-attachments/assets/39917b85-e0a4-40f4-b7b5-d2fa01d04b0b)

![CNOT_Gate_Log_Infidelity_Comparison](https://github.com/user-attachments/assets/eb6f0858-c1ae-421d-a1d7-c116579751b1)


## Conclusions
---
This thesis demonstrated the effectiveness of learning to optimize the control of quantum
gates. The results showed that, although CPPO and CGRPO struggled to achieve stable
learning despite obtaining good fidelities, TD3 performed well in continuous control, and
DDDQN, DPPO and DGRPO were effective in discrete control, albeit limited by the granu-
larity of the action space. The study also highlighted the potential of reinforcement learning
for quantum computing, particularly in optimizing control pulses for quantum gates.
In the future, the integration of machine learning and quantum computing will be essential to advance quantum technologies. Reinforcement learning provides a promising framework for adaptive, hardware-efficient quantum control, paving the way for scalable, and
robust quantum computing

## References
---
[1] M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information.
Cambridge University Press, 10th anniversary ed., 2010.

[2] J. Preskill, “Quantum computing in the nisq era and beyond,” Quantum, vol. 2, p. 79,
2018.

[3] A. Barenco, C. H. Bennett, R. Cleve, D. P. DiVincenzo, N. Margolus, P. Shor,
T. Sleator, J. A. Smolin, and H. Weinfurter, “Elementary gates for quantum com-
putation,” Physical Review A, vol. 52, no. 5, p. 3457, 1995.

[4] P. O. Boykin, T. Mor, M. Pulver, V. Roychowdhury, and F. Vatan, “On universal and
fault-tolerant quantum computing,” arXiv preprint arXiv:quant-ph/9906054v1.

[5] M. Bukov, A. G. Day, D. Sels, P. Weinberg, A. Polkovnikov, and P. Mehta, “Reinforce-
ment learning in different phases of quantum control,” Physical Review X, vol. 8, no. 3,
p. 031086, 2018.

[6] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. MIT Press,
2nd ed., 2018.

[7] P. Palittapongarnpim, P. Wittek, E. Zahedinejad, S. Vedaie, and B. C. Sanders, “Learn-
ing in quantum control: High-dimensional global optimization for noisy quantum dy-
namics,” Neurocomputing, vol. 268, pp. 116–126, 2017.

[8] M. Y. Niu, S. Boixo, V. N. Smelyanskiy, and H. Neven, “Universal quantum control
through deep reinforcement learning,” NPJ Quantum Information, vol. 5, no. 1, p. 33,
2019.

[9] J. Olle, O. M. Yevtushenko, and F. Marquardt, “Scaling the automated discovery of
quantum circuits via reinforcement learning with gadgets,” Mar. 2025. ArXiv preprint
arXiv:2503.11638v1.

[10] M. A. Nielsen and C. M. Dawson, “Universality for quantum computation with partially
defined quantum gates,” arXiv preprint quant-ph/9906054, 1999.

[11] D. d’Alessandro, Introduction to Quantum Control and Dynamics. CRC Press, 2021.

[12] C. W. Duncan, P. M. Poggi, M. Bukov, N. T. Zinner, and S. Campbell, “Taming quan-
tum systems: A tutorial for using shortcuts-to-adiabaticity, quantum optimal control,
& reinforcement learning,” Jan. 2025. ArXiv preprint arXiv:2501.16436v1.

[13] N. V. Vitanov, A. A. Rangelov, B. W. Shore, and K. Bergmann, “Stimulated raman adi-
abatic passage in physics, chemistry, and beyond,” Reviews of Modern Physics, vol. 89,
no. 1, p. 015006, 2017.

[14] M. Goerz, D. Basilewitsch, F. Gago-Encinas, M. G. Krauss, K. P. Horn, D. M. Reich,
and C. Koch, “Krotov: A python implementation of krotov’s method for quantum
optimal control,” SciPost Physics, vol. 7, no. 6, p. 080, 2019.

[15] U. Boscain, M. Sigalotti, and D. Sugny, “Introduction to the Pontryagin maximum
principle for quantum optimal control,” PRX Quantum, vol. 2, no. 3, p. 030203, 2021.

[16] T. H. Su, S. Shresthamali, and M. Kondo, “Quantum framework for reinforcement learn-
ing: integrating markov decision process, quantum arithmetic, and trajectory search,”
Dec. 2024. arXiv prepint arXiv:2412.18208v1.

[17] J. O. Ernst, A. Chatterjee, T. Franzmeyer, and A. Kuhn, “Reinforcement learning for
quantum control under physical constraints,” arXiv preprint arXiv:2501.14372, 2025.

[18] H. Ma, B. Qi, I. R. Petersen, R.-B. Wu, H. Rabitz, and D. Dong, “Machine learn-
ing for estimation and control of quantum systems,” Mar. 2025.arXiv preprint arXiv:2503.03164v1.

[19] P. W. Shor, “Algorithms for quantum computation: discrete logarithms and factoring,”
in Proceedings 35th Annual Symposium on Foundations of Computer Science, pp. 124–
134, IEEE, 1994.

[20] L. K. Grover, “A fast quantum mechanical algorithm for database search,” in Proceed-
ings of the twenty-eighth annual ACM symposium on Theory of computing, pp. 212–219,
ACM, 1996.

[21] J. J. Sakurai and J. Napolitano, Modern Quantum Mechanics. Boston: Addison-Wesley,
2nd ed., 2011.

[22] M. A. Nielsen, “A simple formula for the average gate fidelity of a quantum dynamical
operation,” Physics Letters A, vol. 303, no. 4, pp. 249–252, 2002.

[23] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist rein-
forcement learning,” Machine Learning, vol. 8, no. 3-4, pp. 229–256, 1992.

[24] V. R. Konda and J. N. Tsitsiklis, “Actor-critic algorithms,” in Advances in Neural
Information Processing Systems (NeurIPS), vol. 12, pp. 1008–1014, 1999.

[25] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and
D. Wierstra, “Continuous control with deep reinforcement learning,” arXiv preprint
arXiv:1509.02971, 2015.

[26] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy
optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

115[27] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li,
Y. Wu, and D. Guo, “Deepseekmath: Pushing the limits of mathematical reasoning in
open language models,” arXiv preprint arXiv:2402.03300, 2024.

[28] R.-H. He, H.-D. Liu, S.-B. Wang, J. Wu, S.-S. Nie, and Z.-M. Wang, “Universal quan-
tum state preparation via revised greedy algorithm,” Quantum Sci. Technol., vol. 6,
p. 045021, 2021.

[29] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves,
M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik,
I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-
level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–
533, 2015.

[30] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-
learning,” in Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence
(AAAI), pp. 2094–2100, AAAI Press, 2016.

[31] O. Shindi, Q. Yu, and D. Dong, “A modified deep q-learning algorithm for control
of two-qubit systems,” in 2021 IEEE International Conference on Systems, Man, and
Cybernetics (SMC), (Melbourne, Australia), IEEE, Oct 2021.

[32] O. Shindi, Q. Yu, P. Girdhar, and D. Dong, “A modified deep q-learning algorithm
for optimal and robust quantum gate design of a single qubit system,” in 2022 IEEE
International Conference on Systems, Man, and Cybernetics (SMC), IEEE, 2022.

[33] O. Shindi, Q. Yu, P. Girdhar, and D. Dong, “ Model-Free Quantum Gate Design and
Calibration Using Deep Reinforcement Learning ,” IEEE Transactions on Artificial
Intelligence, vol. 5, no. 01, pp. 346–357, 2024.

[34] S. Fujimoto, H. van Hoof, and D. Meger, “Addressing function approximation error in
actor-critic methods,” in Proceedings of the 35th International Conference on Machine
Learning (ICML), vol. 80, pp. 1587–1596, PMLR, 2018.

[35] H. N. Nguyen, F. Motzoi, M. Metcalf, K. B. Whaley, M. Bukov, and M. Schmitt,
“Reinforcement learning pulses for transmon qubit entangling gates,” Mach. Learn.:
Sci. Technol., vol. 5, p. 025066, 2024.

[36] H. Yu and X. Zhao, “Deep reinforcement learning with reward design for quantum
control,” IEEE Transactions on Artificial Intelligence, vol. 5, p. 1087, Mar 2024.

[37] Z. Wang, N. de Freitas, and M. Lanctot, “Dueling network architectures for deep re-
inforcement learning,” in Proceedings of the 33rd International Conference on Machine
Learning (ICML), vol. 48, pp. 1995–2003, PMLR, 2016.

[38] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in
4th International Conference on Learning Representations (ICLR), 2016.

[39] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, “High-dimensional
continuous control using generalized advantage estimation,” in Proceedings of the 4th
International Conference on Learning Representations (ICLR 2016), 2016.

[40] T.-N. Xu, Y. Ding, J. D. Martı́n-Guerrero, and X. Chen, “Robust two-qubit gate with
reinforcement learning and dropout,” Phys. Rev. A, vol. 110, p. 032614, Sep 2024.

[41] N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, “Optimal
control of coupled spin dynamics: design of nmr pulse sequences by gradient ascent
algorithms,” Journal of Magnetic Resonance, vol. 172, no. 2, pp. 296–305, 2005.

[42] T. Caneva, T. Calarco, and S. Montangero, “Chopped random-basis quantum opti-
mization,” Physical Review A, vol. 84, no. 2, p. 022326, 2011.

[43] T.-N. Xu, Y. Ding, J. D. Martı́n-Guerrero, and X. Chen, “Robust two-qubit gate with
reinforcement learning and dropout,” Physical Review A, vol. 110, p. 032614, 2024.

[44] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin,
N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison,
A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, “Pytorch:
An imperative style, high-performance deep learning library,” in Advances in Neural
Information Processing Systems (NeurIPS), pp. 8024–8035, 2019.

[45] Z. An and D. L. Zhou, “Deep reinforcement learning for quantum gate control,” Euro-
physics Letters (EPL), vol. 126, p. 60002, 2019.

[46] S. Hu, C. Chen, and D. Dong, “Deep reinforcement learning for control design of quan-
tum gates,” in Proceedings of the 13th Asian Control Conference (ASCC 2022), (Jeju
Island, Korea), May 2022.

[47] K.-C. Chen, S. Y.-C. Chen, C.-Y. Liu, and K. K. Leung, “Quantum-train-based dis-
tributed multi-agent reinforcement learning,” 2024. arXiv prepint arXiv:2412.08845.

[48] A. Dubal, D. Kremer, S. Martiel, V. Villar, D. Wang, and J. Cruz-Benito, “Pauli
network circuit synthesis with reinforcement learning,” 2025. arXiv preprint arXiv:
2503.14448.

[49] Z. Wang, C. Feng, C. Poon, L. Huang, X. Zhao, Y. Ma, T. Fu, and X.-Y. Liu, “Re-
inforcement learning for quantum circuit design: Using matrix representations,” 2025.
arXiv preprint arVix:2501.16509.

[50] P. Altmann, J. Stein, M. Kölle, A. Bärligea, T. Gabor, T. Phan, S. Feld, and
C. Linnhoff-Popien, “Challenges for reinforcement learning in quantum circuit design,”
2024. arXiv preprint arVix: 2312.11337.

[51] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, “Surface codes:
Towards practical large-scale quantum computation,” Physical Review A, vol. 86,
p. 032324, 2012.

[52] A. Steane, “Error correcting codes in quantum theory,” Physical Review Letters, vol. 77,
no. 5, p. 793, 1996.

[53] Z. T. Wang, Q. Chen, Y. Du, Z. H. Yang, X. Cai, K. Huang, J. Zhang, K. Xu, J. Du,
Y. Li, Y. Jiao, X. Wu, W. Liu, X. Lu, H. Xu, Y. Jin, R. Wang, H. Yu, and S. P.
Zhao, “Quantum compiling with reinforcement learning on a superconducting proces-
sor,” 2024. arXiv preprint arXiv: 2406.12195.

[54] I. Khalid, C. A. Weidner, E. A. Jonckheere, S. G. Schirmer, and F. C. Langbein,
“Sample-efficient model-based reinforcement learning for quantum control,” Physical
Review Research, vol. 5, p. 043002, 2023.

[55] Y. Chow, O. Nachum, E. Duenez-Guzman, and M. Ghavamzadeh, “A lyapunov-based
approach to safe reinforcement learning,” in Proceedings of the 32nd Conference on
Neural Information Processing Systems (NeurIPS 2018), (Montréal, Canada), 2018.

[56] S. C. Hou and X. X. Yi, “Quantum lyapunov control with machine learning,” Quantum
Information Processing, vol. 19, no. 8, 2020.

[57] Y. Baum, M. Amico, S. Howell, M. Hush, M. Liuzzi, P. Mundada, T. Merkh, A. R.
Carvalho, and M. J. Biercuk, “Experimental deep reinforcement learning for error-
robust gate-set design on a superconducting quantum computer,” PRX Quantum, vol. 2,
p. 040324, Nov 2021.

[58] Q. Chen, Y. Du, Y. Jiao, X. Lu, X. Wu, and Q. Zhao, “Efficient and practical quantum
compiler towards multi-qubit systems with deep reinforcement learning,” Quantum Sci.
Technol., vol. 9, p. 045002, 2024.

[59] D. Koutromanos, D. Stefanatos, and E. Paspalakis, “Control of qubit dynamics using
reinforcement learning,” Information, vol. 15, no. 5, p. 272, 2024.

[60] M. C. Smith, A. D. Leu, K. Miyanishi, M. F. Gely, and D. M. Lucas, “Single-qubit
gates with errors at the 10−7 level,” 2024. arXiv preprint arXiv:2412.04421.

[61] A. Javadi-Abhari, M. Treinish, K. Krsulich, C. J. Wood, J. Lishman, J. Gacon, S. Mar-
tiel, P. D. Nation, L. S. Bishop, A. W. Cross, B. R. Johnson, and J. M. Gambetta,
“Quantum computing with qiskit,” 2024. arXiv preprint arXiv: 2405.08810.

[62] D. Koutromanos, “Qubit control with reinforcement learning methods,” Master’s thesis,
Democritus University of Thrace, Department of Electrical and Computer Enginee
