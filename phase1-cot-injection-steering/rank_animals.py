"""Rank a candidate animal vocabulary by English corpus frequency (wordfreq zipf)."""
from wordfreq import zipf_frequency

CANDIDATES = """
dog cat horse bird fish cow pig sheep goat chicken duck goose rabbit mouse rat
lion tiger bear wolf fox deer elephant giraffe zebra monkey ape gorilla chimpanzee
snake lizard frog toad turtle tortoise crocodile alligator dinosaur dragon
eagle hawk owl falcon crow raven pigeon dove sparrow robin parrot penguin ostrich
swan peacock flamingo pelican seagull woodpecker hummingbird vulture stork
whale dolphin shark octopus squid jellyfish crab lobster shrimp starfish seal
walrus otter beaver dolphinfish salmon trout tuna cod eel stingray seahorse
bee wasp ant spider butterfly moth beetle fly mosquito grasshopper cricket
dragonfly ladybug caterpillar worm snail slug scorpion centipede termite
cheetah leopard jaguar panther lynx cougar puma hyena jackal coyote
panda koala kangaroo wallaby platypus wombat possum sloth armadillo anteater
camel llama alpaca donkey mule ox bull calf lamb pony
squirrel chipmunk hedgehog porcupine badger raccoon skunk weasel ferret mole
bat hamster guinea gerbil chinchilla
buffalo bison moose elk antelope gazelle impala rhinoceros hippopotamus
seahorse manatee narwhal orca porpoise
turkey quail pheasant partridge chick rooster hen
lemur baboon orangutan gibbon macaque marmoset
cobra python viper rattlesnake anaconda boa gecko iguana chameleon salamander
newt tadpole
mammoth sabertooth
mink sable otterhound
wildcat bobcat ocelot serval caracal
meerkat mongoose civet
warthog boar hog piglet
reindeer caribou yak ibex chamois
albatross puffin kingfisher heron egret crane ibis toucan cardinal finch
canary blackbird thrush swallow swift nightingale magpie jay starling
mackerel herring sardine anchovy carp catfish perch pike bass halibut
clam oyster mussel scallop urchin anemone coral sponge
mantis termite aphid locust cicada firefly weevil
llama vicuna guanaco
""".split()

seen, cands = set(), []
for w in CANDIDATES:
    if w not in seen:
        seen.add(w)
        cands.append(w)

ranked = sorted(((zipf_frequency(w, "en"), w) for w in cands), reverse=True)
top100 = [w for z, w in ranked[:100]]

print(f"{len(cands)} candidates -> top 100 by zipf\n")
for i, (z, w) in enumerate(ranked[:100], 1):
    print(f"{i:>3}. {w:<14} zipf={z:.2f}")

print("\nANIMALS100 = " + repr(top100))
