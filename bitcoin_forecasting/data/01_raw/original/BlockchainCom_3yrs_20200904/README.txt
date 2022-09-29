https://www.blockchain.com/charts#currency

0-3: Currency Statistics
4-9: Block Details
10-17: Mining Information
18-28: Network Activity
29-31: Market Signals


0
The total number of mined bitcoin that are currently circulating on the network.

Explanation
The total supply of BTC is limited and pre-defined in the Bitcoin protocol at 21 million, with the mining reward (how Bitcoins are created) decreasing over time. This graph shows how many Bitcoins have already been mined or put in circulation.
Notes
The Bitcoin reward is divided by 2 every 210,000 blocks, or approximately four years. Some of the Bitcoins in circulation are believed to be lost forever or unspendable, for example because of lost passwords, wrong output addresses or mistakes in the output scripts.
Methodology
The number of Bitcoins in circulation is calculated from the theoretical reward defined by the Bitcoin protocol.

1
The average USD market price across major bitcoin exchanges.

Explanation
The market price is how much you can sell 1 Bitcoin (BTC) for. The supply of BTC is limited and pre-defined in the Bitcoin protocol. This means that the price is sensitive to shifts in both supply and demand. In total, 21 Millions BTC can be mined and the Total Circulating Bitcoin chart displays how many of them have already been found.
Notes
The smallest amount of BTC that somebody can own is 1 Satoshi, and there are 100,000,000 Satoshi in 1 BTC. This means that it is possible to buy and sell fractions of a Bitcoin.
Methodology
The market price is a consolidation of prices from crypto exchanges market data.

2
The total USD value of bitcoin in circulation.

3
The total USD value of trading volume on major bitcoin exchanges.

Explanation
The Bitcoin trading volume indicates how many Bitcoins are being bought and sold on specific exchanges. High trading volumes are likely to drive more on-chain activity, for example when people deposit and withdraw funds. It is also a good indicator of the general interest in the crypto market.
Notes
The displayed volume is only from a small proportion of exchanges. The actual total trading volume is much higher. Part of the trading volume is also made outside crypto exchanges, for example in the OTC (Over The Counter) market.
Methodology
The trading volume is the sum of the trading volume of the BTC/USD pair from some exchanges.

4
The total size of the blockchain minus database indexes in megabytes.

5
The average block size over the past 24 hours in megabytes.

6
The average number of transactions per block over the past 24 hours.

7
The total number of transactions on the blockchain.

8
The median time for a transaction with miner fees to be included in a mined block and added to the public ledger.

9
The average time for a transaction with miner fees to be included in a mined block and added to the public ledger.

10
The estimated number of terahashes per second the bitcoin network is performing in the last 24 hours.

Explanation
Mining hashrate is a key security metric. The more hashing (computing) power in the network, the greater its security and its overall resistance to attack. Although Bitcoin’s exact hashing power is unknown, it is possible to estimate it from the number of blocks being mined and the current block difficulty.
Notes
Daily numbers (raw values) may periodically rise or drop as a result of the randomness of block discovery : even with a hashing power constant, the number of blocks mined can vary in day. Our analysts have found that looking at a 7 day average is a better representation of the underlying power.
Methodology
The hashing power is estimated from the number of blocks being mined in the last 24h and the current block difficulty. More specifically, given the average time T between mined blocks and a difficulty D, the estimated hash rate per second H is given by the formula H = 2 32 D / T

11
A relative measure of how difficult it is to mine a new block for the blockchain.

Explanation
The difficulty is a measure of how difficult it is to mine a Bitcoin block, or in more technical terms, to find a hash below a given target. A high difficulty means that it will take more computing power to mine the same number of blocks, making the network more secure against attacks. The difficulty adjustment is directly related to the total estimated mining power estimated in the Total Hash Rate (TH/s) chart.
Notes
The difficulty is adjusted every 2016 blocks (every 2 weeks approximately) so that the average time between each block remains 10 minutes.
Methodology
The difficulty comes directly from the confirmed blocks data in the Bitcoin network.

12
Total value of coinbase block rewards and transaction fees paid to miners.

13
The total BTC value of all transaction fees paid to miners. This does not include coinbase block rewards.

14
The total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.

15
Average transaction fees in USD per transaction.

16
A chart showing miners revenue as percentage of the transaction volume.

17
A chart showing miners revenue divided by the number of transactions.

18
The total number of unique addresses used on the blockchain.

19
The total number of confirmed transactions per day.

Explanation
The number of daily confirmed transactions highlights the value of the Bitcoin network as a way to securely transfer funds without a third part.
Notes
Transactions are accounted for only once they are included in a block. During times of peak mempool congestion, transactions with lower fees are likely to be confirmed after a few hours or even days in rare cases. While this graph is a suitable medium and long term indicator, the Mempool Size (Bytes) and Mempool Transaction Count charts are more suitable for short term network activity.
Methodology
Transactions from confirmed blocks are simply summed up to obtain daily numbers

20
The number of transactions added to the mempool per second.

21
The total value of all transaction outputs per day. This includes coins returned to the sender as change.

22
The total number of unconfirmed transactions in the mempool.

Explanation
The mempool is where all valid transactions wait to be confirmed by the Bitcoin network. A high number of transactions in the mempool indicates a congested traffic which will result in longer average confirmation time and higher priority fees. The mempool count metric tells how many transactions are causing the congestion whereas the Mempool Size (Bytes) chart is a better metric to estimate how long the congestion will last.
Notes
In order to be confirmed, a transaction from the mempool needs to be included in a block. Unlike the maximum size of a block which is fixed, the maximum number of transactions which can be included in a block varies, because not all transactions have the same size.
Methodology
Each Bitcoin node builds its own version of the mempool by connecting to the Bitcoin network. The mempool content is aggregated from a few instances of up to date Bitcoin nodes maintained by the Blockchain.com engineering team; this way, we gather as much information as possible to provide accurate mempool metrics.

23
The rate at which the mempool is growing in bytes per second.


24
The aggregate size in bytes of transactions waiting to be confirmed.

Explanation
The mempool is where all the valid transactions wait to be confirmed by the Bitcoin network. A high mempool size indicates more network traffic which will result in longer average confirmation time and higher priority fees. The mempool size is a good metric to estimate how long the congestion will last whereas the Mempool Transaction Count chart tells us how many transactions are causing the congestion.
Notes
In order to be confirmed, a transaction from the mempool needs to be included in a block. The size of a block cannot exceed 4 million weight units (1 million vbytes) , and each transaction has its own weight depending on the type of transaction, the UTXOs it spends (inputs) and the addresses it sends to (outputs).
Methodology
Each Bitcoin node builds its own version of the mempool by connecting to the Bitcoin network. The mempool content is aggregated from a few instances of up to date Bitcoin nodes maintained by the Blockchain.com engineering team; this way, we gather as much information as possible to provide accurate mempool metrics.

25
The total number of valid unspent transactions outputs. This excludes invalid UTXOs with opcode OP_RETURN

26
The total number of transactions excluding those involving the network's 100 most popular addresses.

27
The total estimated value in BTC of transactions on the blockchain. This does not include coins returned as change.

28
The total estimated value in USD of transactions on the blockchain. This does not include coins returned as change.

29
MVRV is calculated by dividing Market Value by Realised Value. In Realised Value, BTC prices are taken at the time they last moved, instead of the current price like in Market Value.

30
NVT is computed by dividing the Network Value (= Market Value) by the total transactions volume in USD over the past 24hour.

31
NVTS is a more stable measure of NVT, with the denominator being the moving average over the last 90 days of NVT's denominator.


