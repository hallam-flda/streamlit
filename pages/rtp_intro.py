import streamlit as st

st.title("Return To Player Inefficiency")

st.markdown(
"""
It is standard on online Casinos to advertise the Return to Player (RTP) of each game. This is premised on the player playing a large number of games with unbounded tolerance for
volatile swings in balance. I want to investigate how fair these RTP's are and whether introducing some reasonable constraints with respect to time played or budget will affect
the quoted RTP.
"""
)

st.header("Example of Quoted RTP", divider = True)

st.markdown(
"""
Live Casino games are the closest you can get to playing in a physical casino. The rules are the same and the table is operated by a human, live-streamed to the user who plays online.
 Since gambling is a highly regulated industry, we can assume the table is fair and the rules follow the same as a standard casino. The RTP for any selection on European Roulette is 97.3%
 the proof of which can be found [here](https://hallam-flda.streamlit.app/stochastic_processes).

From the [stake.com](https://stake.com/casino/games/evolution-roulette-lobby) website, we can see an example of how this RTP is quoted on an online casino.
"""    
)

st.image("media/rtp/roulette_rtp_example.png")

st.header("What's the Issue?", divider = True)

st.markdown(
"""
Roulette is a simple game mathematically and many players clearly understand the house edge and the implications this has on their play. Where this isn't as clear is online slots where the maths is
obfuscated behind enticing bonus mechanisms. The game in the video below is quoted to have an RTP of 96.0% which implies you can expect to leave the game with a similar position compared to playing European Roulette.
However, there is no qualification of the meaning behind the game's 'high' volatility.

If you watch the spins below, it is far harder to calculate the underlying rules of the game. I would argue this leads to a discrepancy in consumer expectations with regards to RTP. Clearly it is not sensible to promote the message
that consumers should tolerate spells of volatility because in the long run they will return to the average, but is there a fairer way to represent RTPs?
"""    
)

st.video("media/rtp/high_vola_slot.mp4")

with st.expander("See Game Information"):
    st.image("media/rtp/high_vol_game_info.png")