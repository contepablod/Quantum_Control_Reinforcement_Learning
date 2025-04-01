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

First, I would like to express my deepest gratitude to my advisor, Prof. Emmanuel Pas-
palakis, for his invaluable guidance, encouragement, and unwavering support throughout
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
