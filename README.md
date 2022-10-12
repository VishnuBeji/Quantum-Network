# Quantum-Network

This paper considers the problem of secure packet routing at the maximum achievable rate in a Quantum key distribution (QKD) network. Assume that a QKD protocol generates symmetric private keys for secure communication over each link in a multi-hop network. The quantum key generation process, which is affected by noise, is assumed to be modeled by a stochastic counting process. Packets are first encrypted with the available quantum keys for each hop and then transmitted on a point-to-point basis over the communication links. A fundamental problem that arises in this setting is to design a secure and capacity-achieving routing policy that accounts for the time-varying availability of the quantum keys for encryption and finite link capacities for transmission. In this paper, by combining the QKD protocol with the Universal Max Weight (UMW) routing policy, we design a new secure throughput-optimal routing policy, called Tandem Queue Decomposition (TQD). TQD solves the problem of secure routing efficiently for a wide class of traffic, including unicast, broadcast, and multicast. One of our main contributions in this paper is to show that the problem can be reduced to the usual generalized network flow problem on a transformed network without the key availability constraints. Simulation results show that the proposed policy incurs a substantially smaller delay as compared to the state-of-the-art routing and key management policies. The proof of throughput-optimality of the proposed policy makes use of the Lyapunov stability theory along with a careful treatment of the key-storage dynamics.

We use ford fulkerson algorithm to find network layer capacity region. Then we use Djikstras to find shortest weighted path.

The improvements we make:
- `Delay Performance`: A better delay performance as compared to Back-Pressure based solutions.
- `Cycle Hopping`: Avoids cycle hopping as paths are computed upon arrival of packets.

The results: //

`Unicast`

<img src="https://github.com/VishnuBeji/Quantum-Network/blob/master/images/fireee.png" width="250px"/>

`Broadcast` 

<img src="https://github.com/VishnuBeji/Quantum-Network/blob/master/images/broadcastSim.png" width="250px"/>

Link to papaer can be found [here](https://ieeexplore.ieee.org/abstract/document/9668578)
