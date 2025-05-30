import random
from openai import OpenAI
from datasets import Dataset, load_dataset
from datetime import datetime


class MrlPromptCreator:
    def __init__(self, examples: dict[int, str] = None, long_examples: dict[int, str] = None):
        self.example_strs = {
            1: '''
      [
        (
          ("Can you tell me a story about the tallest mountain in the world and the brave people who try to climb it?",
          """
          High in the kingdom of clouds stands Mount Everest, wearing a forever-white snow cloak. Adventurers spend 2 months climbing its icy shoulders.
          First they walk through green valleys with yaks carrying tents. At Base Camp (5,300m), they wait 3 weeks for their bodies to grow used to thin air.
          The dangerous Khumbu Icefall comes next - climbers cross crackling blue ice bridges using metal ladders. Sherpa guides fix ropes up the Western Shoulder
          where winds howl like dragons. Near the top (8,850m), everyone wears oxygen masks because the air has only 1/3 of normal breath. At 5am, the final push
          to the summit - freezing toes, burning lungs, but hearts full of joy when they stand on the roof of the world!
          """),
          [("What special clothes do Everest climbers wear at the top?",
            """
            At Everest's peak, climbers wear 5 layers: wool socks like bear paws, heated boots, down suits fluffier than baby chicks, mittens bigger than dinner plates,
            and masks feeding warm oxygen. Their goggles have gold mirrors to reflect the blinding white snow. Even dressed like snowmen, fingers still turn to icicles
            after 30 minutes in the -40°C freeze!
            """)]
        ),
        (
          ("Could you explain why the Nile River was so important to ancient civilizations and how it helped them live?",
          """
          The Nile was Egypt's watery life-road. Every summer, like clockwork, it flooded its banks for 100 days. When the waters retreated, they left black mud perfect
          for growing wheat and papyrus. Farmers called this rich soil "Black Land" and the desert "Red Land". Boats carried stone blocks for pyramids down the river's
          north-flowing current. At night, people watched the Nile mirror the stars - they believed it flowed from the heavens to feed their fields. Without the Nile's
          gifts, there would be no pharaohs, pyramids, or hieroglyph stories!
          """),
          [("How did the Nile's floods help grow food?",
            """
            The Nile's summer flood was nature's perfect farmer! First, water soaked dry earth for 100 days. Next, it left mud full of plant food from faraway mountains.
            Farmers planted seeds in this black gold, using river water in clay canals. Wheat grew tall, papyrus became paper, and melons swelled sweet. Even fish swam
            into traps when waters receded - the Nile fed bellies, fed animals, and fed Egypt's greatness!
            """)]
        ),
        (
          ("What's the journey of honey from flower to jar? Tell me like a bee's adventure!",
          """
          Buzzbee Ziggy starts at sunrise, visiting 1,000 flowers before lunch! Her tongue-straw slurps nectar into a special honey stomach. Back at the hexagon castle,
          she vomits the nectar into Waxworker Molly's mouth. Molly chews it with enzyme magic, then spits it into honeycomb cells. Wing-fan bees hover like helicopters,
          blowing air to thicken the nectar. After 3 days of fanning, the cells get wax lids. Beekeepers later take extra honeycombs, spin them in metal drums, and golden
          rivers flow into jars. One jar holds the work of 300 bees' whole lives!
          """),
          [("Why do bees fan the honeycombs?",
            """
            Fanning is the bee air conditioner! Wings beat 200 times/second to blow warm air over honey soup. This makes water evaporate until honey gets thick and sticky.
            Too runny? Mold grows. Too thick? Can't pour. Perfect fanning makes honey last 3,000 years - jars in pharaoh tombs still taste sweet! Bees take turns fanning,
            resting wings when tired. Their teamwork makes perfect breakfast honey!
            """)]
        ),
        (
          ("Describe a day in the life of a rainforest tree from seed to canopy",
          """
          Little Kapok seed sprouts in dark soil, drinking rotten leaf soup. For 5 years, it grows 2cm/month in the Understory's twilight. Year 6 - a storm kills a giant
          tree, making a sun hole! Kapok shoots up 3m in months, growing stilt roots like spider legs. Ants move into its thorns, biting leaf-eaters. By year 20, it reaches
          the Canopy layer (30m high), growing bat-pollinated flowers. Year 50 - it's an Emergent giant (60m)! Its cotton-like seeds float for miles, starting new adventures.
          """),
          [("How do Kapok trees grow so fast when there's sunlight?",
            """
            Kapoks are sunlight hoarders! When a storm makes a forest hole, they soak up 12x more energy than normal. Special pipes in their trunk pump water 3 stories high.
            Leaves grow dinner-plate size to catch every sunbeam. They even steal nutrients from nearby roots! This growth spurt lets them reach the sunny canopy before the
            hole closes - a race against time and shadows!
            """)]
        ),
        (
            ("How do volcanoes make new islands in the ocean? Tell it like an underground giant's recipe!",
            """
            Deep under the sea, Earth's kitchen has a fiery oven called magma. When the oven gets too hot (over 1,000°C!), bubbly rock soup pushes through cracks.
            This lava meets cold water and freezes into black pillow shapes. After 500 eruptions over 1,000 years, the pillows pile up like rocky pancakes.
            Finally, one day the island peeks above waves! Baby plants arrive on bird feet. Coconut boats wash ashore. Soon the new island is ready for palm trees
            and crabs! Hawaii grew this way - 137 islands from underwater cooking!
            """),
            [("What happens when lava meets seawater?",
              """
              Lava vs water is like hot candy hitting snow! The outside instantly hardens into black glassy pillows while inside stays molten. Steam explosions make
              popping sounds! Over time, these pillowy rocks build island foundations. Divers find whole fields of these lava pillows around new islands!
              """)]
        ),
        (
            ("Describe the amazing journey of monarch butterflies across countries",
            """
            Every fall, millions of orange-black monarchs leave Canada like living confetti. They fly 4,000 km to Mexico's oyamel forests, riding air currents like
            invisible rivers. No single butterfly knows the way - it's in their antennae GPS! They cluster on fir trees, turning branches into orange fur. After winter,
            great-grandchildren return north, laying eggs on milkweed. It takes 4 generations to complete the cycle - nature's relay race with wings!
            """),
            [("How do baby monarchs know where to go?",
              """
              New monarchs inherit sky maps in their tiny brains! They use sun position like a compass and feel Earth's magnetic fields through their antennae.
              Smell memories from caterpillar days help find milkweed. Though they've never been to Mexico, their wings remember the ancient path!
              """)]
        ),
        (
            ("What makes the seasons change? Tell me about Earth's tilted hat adventure!",
            """
            Earth wears a invisible tilted hat (23.5°) as it circles the sun. When the North Pole tips toward sun - summer! Long days, short shadows.
            6 months later, the South Pole gets sun's attention - winter here! Spring and autumn happen when Earth's hat isn't tipping too far either way.
            This tilt makes leaf colors, animal sleeps, and snowball fights possible. Without the tilt, every day would be same weather - how boring!
            """),
            [("Why do leaves fall in autumn?",
              """
              Trees throw a color party before winter sleep! As days shorten, they stop making green chlorophyll. Hidden yellow-orange colors shine through.
              A special cork layer grows where leaves attach - like Band-Aid that can't stick forever. Wind whispers "Let go!" and leaves dance down to become
              next year's soil food. The tree sleeps until spring's sun alarm!
              """)]
        ),
        (
            ("Tell me a story about camels surviving in hot deserts without water",
            """
            Sandy the camel hasn't drunk in 2 weeks! Her secret? Hump fuel! The hump stores 36kg of fat - when needed, her body turns this into water (and energy!).
            Thick fur keeps sun off her skin like a umbrella. She breathes through nose wrinkles that catch moisture. Even her blood cells are oval-shaped to keep
            flowing when dehydrated. At night, her temperature drops to 34°C to save energy. Camels are desert survival superheroes!
            """),
            [("How do camel humps really work?",
              """
              Humps aren't water balloons - they're energy banks! 1kg of hump fat makes 1 liter of water when broken down. The process needs oxygen, so camels
              breathe slowly. Their red blood cells stretch like accordions to survive thick blood. Baby camels drink 20 liters in 10 minutes - filling their
              future hump fuel!
              """)]
        ),
        (
            ("Why are coral reefs called 'rainforests of the sea'? Describe a reef's busy day",
            """
            Dawn on the reef! Purple coral polyps stretch sticky arms to catch breakfast plankton. Clownfish dance in anemone wigs. A parrotfish crunches coral
            for lunch, pooping white sand. Cleaner shrimp set up stations - "Free teeth brushing!" they signal. Hawksbill turtles munch sponges. At night,
            corals show glowing tips while octopuses hunt. Every creature has a job - building, cleaning, eating, being eaten. More species than a jungle -
            25% of sea life lives here!
            """),
            [("What happens when corals get too warm?",
              """
              Heat makes corals vomit their colorful algae roommates! This is bleaching - without algae, corals turn white and hungry. If cool water returns quickly,
              algae move back in. But long heatwaves leave dead ghost reefs. Fish schools disappear. Seaweed takes over. Protecting reefs means keeping ocean
              temperatures just right!
              """)]
        ),
        (
            ("Explain how the moon changes shape in the sky each night",
            """
            The moon plays peek-a-boo with Earth's shadow! As it orbits us every 29 days, sunlight hits it from different angles. When between Earth and sun -
            New Moon (invisible!). A week later - right side glows (First Quarter). Full Moon shows its whole face when opposite sun. Then it wanes left side.
            The phases helped ancient farmers track time. Moonlight is really sun's glow bouncing off moon dust - no fire inside, just space mirror!
            """),
            [("Why do we always see the same moon face?",
              """
              The moon does a slow spin dance! It rotates exactly once per Earth orbit - like ballerina keeping face towards partner. This "tidal locking" happened
              over millions of years. The far side has more craters but we never see it from Earth. Astronauts who circled moon saw the hidden face - mountains
              and plains without seas!
              """)]
        ),
        (
            ("Describe the journey of a tornado from first wind to final spin",
            """
            It starts when hot and cold air argue high above! Warm moist air rises fast, forming thunderclouds. Winds at different heights blow opposite directions -
            like rubbing hands to make heat. A horizontal spinning tube forms. Rising air tilts it vertical - hello tornado! The funnel reaches down, sucking up
            dirt and debris. For 10 violent minutes, it destroys everything in its 300m wide path. Then rain cools the argument, and the tornado dies.
            Storm's over - time to rebuild.
            """),
            [("Where's the safest place during a tornado?",
              """
              Underground is best - storm cellars or basements. No basement? Go to windowless inner room like bathroom. Cover with mattresses! Cars are dangerous
              - they can fly. If outside, find ditch and protect head. Tornadoes sound like freight trains. After passing, be careful - new ones might form from
              same storm!
              """)]
        ),
        (
            ("How do seeds travel to new homes? Tell me their adventure stories!",
            """
            Maple Samara jumps from branch - her paper wings spin like helicopter! She lands 200m from mom. Coconut Corky floats 4,000km on ocean waves to tropical
            beaches. Burr brothers hitchhike on deer fur. Poppy Pod shakes like pepper shaker in wind. Ants carry elaiosome snacks, dropping seeds in new soil.
            Some seeds wait 100 years for fire to crack their shells. Every seed has a travel plan - some quick, some slow, all hoping for perfect home!
            """),
            [("Which seed uses animal taxis?",
              """
              Sticky seeds like burdock use Velcro fur rides! Their hooks cling to fox fur or hiking socks. Some sweet seeds bribe ants with oil treats.
              Jungle seeds hide in tasty fruit - animals eat them and poop seeds miles away. Even fish carry seeds stuck to their scales! Seeds are nature's
              best hitchhikers!
              """)]
        ),
        (
            ("What do bears do during winter sleep? Describe a hibernation cabin",
            """
            Brownie Bear eats 20,000 berries to build fat blanket! In October, she finds a cozy cave - the Hibernation Hotel. Her heartbeat slows from 50 to 8 beats
            per minute. Body temperature drops 10°C. She doesn't poop for months! Every 2 weeks, she shivers awake for 1 hour, then back to sleep. Baby cubs are
            born tiny (like squirrels!) during this sleep, nursing while mom dreams. In spring, 100kg lighter but rested, she emerges ready for honey!
            """),
            [("How do bears stay healthy without moving?",
              """
              Magic hibernation blood! Their bodies recycle waste into protein. Special hormones keep muscles strong without exercise. Thick fur and fat prevent
              freezing. Slow breathing saves oxygen. Even their bones stay strong! Scientists study bear blood to help astronauts on long space trips - maybe
              future humans will hibernate too!
              """)]
        ),
        (
            ("Why do northern lights dance in the sky? Tell me their colorful show story",
            """
            Solar wind (charged space particles) races toward Earth at 1 million mph! Earth's magnetic shield guides them to the poles. Here, they crash into
            air molecules 100km up. Oxygen glows green-red, nitrogen blue-purple. The lights swirl like giant curtains in solar wind breezes. Best shows happen
            during strong solar storms. Ancient people thought they were warrior spirits - we know it's Earth's nightlight show with space electricity!
            """),
            [("What makes different aurora colors?",
              """
              Sky chemistry class! 100-300km high: oxygen glows yellow-green. Higher up (300-400km), oxygen does rare red dance. Nitrogen molecules at 100km
              flash blue skirts when hit. Mix them for purple! The colors show what air is doing up there. Strong solar storms make the whole sky ripple like
              rainbow flags!
              """)]
        )
      ]
      ''',
            2: '''
      [
        (
            ("Tell me a story about climbing Mount Everest",
            """
            Mount Everest is the world's tallest mountain (8,850m). Climbers spend 2 months slowly climbing up.
            They face thin air, -40°C cold, and dangerous ice cracks. Sherpa guides fix ropes through the Khumbu Icefall's
            moving glaciers. At the top, you can see Earth's curve and need bottled oxygen to breathe!
            """),
            [
                ("How long does preparation take before summit day?",
                """
                Climbers wait 3 weeks at Base Camp (5,300m) for their bodies to grow extra blood cells. They practice
                crossing ice bridges with spiked boots. Yaks carry tents and food up rocky trails. Doctors check everyone's
                health daily - only the strongest get summit permission!
                """),
                ("What's the most dangerous part after Base Camp?",
                """
                The Khumbu Icefall! Towering ice walls crack and shift daily. Climbers wake at 3am when ice is frozen solid.
                They sprint through using aluminum ladders over bottomless cracks. Sherpas repair the path daily - one wrong
                step and...CRASH!
                """)
            ]
        ),
        (
            ("Describe how the Nile River helped build pyramids",
            """
            The Nile flooded every July, leaving perfect mud for brick-making. Workers floated giant stone blocks on reed boats.
            They built ramps from river clay. At night, stars reflected on water guided pyramid alignment. Without the Nile's
            gifts, Egypt's wonders wouldn't exist!
            """),
            [
                ("How did floods help make pyramid bricks?",
                """
                Floodwaters carried black soil called 'kemet.' Workers mixed this with straw and poured into wooden molds.
                Sun-baked bricks hardened in 3 days. The best bricks lined pyramid cores - 2.3 million blocks per pyramid!
                """),
                ("Why were stars important for pyramid builders?",
                """
                Builders used the North Star (always visible) to align pyramids perfectly north-south. They mirrored star
                patterns on the Nile's reflections. Secret tunnels inside pyramids pointed to Sirius, the flood-prediction star!
                """)
            ]
        ),
        (
            ("How do monarch butterflies know where to migrate?",
            """
            Monarchs born in fall have super-sized wings and antennae GPS. They fly 4,000km to Mexico's oyamel forests using
            sun position and Earth's magnetism. Their great-grandchildren return north next spring - a four-generation
            round trip!
            """),
            [
                ("What's special about fall-born monarchs?",
                """
                Fall monarchs live 8 months (vs 2-5 weeks)! They store fat like tiny fuel tanks and sense cold fronts.
                Their wings are darker for sun warmth. They even drink flower nectar mid-flight without landing!
                """),
                ("How do baby monarchs find milkweed?",
                """
                Moms lay eggs only on milkweed leaves. Caterpillars memorize the smell! Adults use foot sensors to test plants.
                They also spot milkweed's pink flowers from 10m away - nature's bullseye!
                """)
            ]
        ),
        (
            ("Explain how volcanoes create new islands",
            """
            Underwater volcanoes erupt for thousands of years. Lava cools into black pillow shapes. Over centuries, these
            pile up until an island peeks above waves. Waves grind rocks into white sand. Birds bring seeds in feathers - soon
            palm trees sway on new land!
            """),
            [
                ("What do underwater eruptions look like?",
                """
                Red-hot lava meets cold ocean = instant steam explosions! Black smoke plumes rise while molten rock forms
                blob-shaped pillows. Deep-sea cameras show ghostly shrimp dancing around warm vents!
                """),
                ("How do plants reach new islands?",
                """
                Coconut shells float 4,000km on currents. Bird feet carry sticky seeds. Storms blow light spores. Mangrove
                seeds sprout while still on parent trees! First plants are pioneers - their roots break rock into soil.
                """)
            ]
        ),
        (
            ("Tell me about beavers building dams",
            """
            Beaver families work night shifts using sharp teeth. They cut trees upstream, float logs to dam sites, and weave
            them with mud. Dams create ponds for safe lodges. The biggest dam (850m) was seen from space - animal engineers!
            """),
            [
                ("Why do beavers need ponds?",
                """
                Ponds protect lodges from wolves and bears. Underwater lodge entrances stay ice-free in winter. Stored food
                (branches) stays fresh underwater. Fish attract otters who chase away enemies - smart neighbors!
                """),
                ("How do beaver teeth stay sharp?",
                """
                Beaver teeth have iron in front (orange color!) making them chisel-hard. Back teeth grind wood into pulp.
                They grow continuously - chewing files them down. No dentist needed!
                """)
            ]
        ),
        (
          ("How do humpback whales hunt in teams? Tell me their bubble net story!",
          """
          Humpback whales in Alaska use bubble nets to catch fish feasts! One whale blows bubbles in a spiral circle while others
          sing loud songs. The bubbles act like a fishing net - fish panic and ball up. Then whales surge up with mouths open,
          swallowing 500kg of fish in one gulp! They take turns being bubble blowers and singers - true underwater orchestras!
          """),
          [
              ("Why do whales sing during bubble feeding?",
              """
              The songs make fish swim downward into the bubble trap! Low notes vibrate fish swim bladders, confusing them.
              High notes keep the whale team in sync. Each pod has unique songs passed down through generations - a musical
              fishing tradition!
              """),
              ("How do baby whales learn bubble net fishing?",
              """
              Calves watch moms for 2 years before trying. First they practice blowing weak bubbles that pop too fast.
              Teens make messy spirals that fish escape from. Adults correct them with nudge-and-show lessons. By age 5,
              they're master bubble net chefs!
              """)
          ]
        ),
        (
            ("Explain how hurricanes get so strong over the ocean",
            """
            Hurricanes are giant heat engines! They start as storm clusters over 26°C water. Warm moist air rises like a
            chimney, creating low pressure. Winds spiral inward, getting faster like ice skater pulling arms in. The eye forms
            when spinning hits 120km/h - calm center surrounded by deadly walls of storm. One hurricane can release 200 times
            the world's daily electricity!
            """),
            [
                ("Why does the hurricane eye feel calm?",
                """
                The eye is the storm's spinning balance point! Air sinks here, creating a dry zone with light winds. But it's
                temporary - eye walls with 250km/h winds surround it. The calm lasts 30-60 minutes before violent winds return
                from opposite direction!
                """),
                ("How does warm ocean water fuel hurricanes?",
                """
                Hurricanes drink heat like giant straws! Every second, they absorb ocean heat equal to 10 atomic bombs.
                This powers the evaporation-condensation engine. Cooler water or land breaks the cycle - that's why storms
                weaken after landfall.
                """)
            ]
        ),
        (
            ("Describe how sequoia trees grow so tall and old",
            """
            Giant sequoias are Earth's tallest trees (95m)! Their secret: fire-resistant bark thick as pizza dough. Roots spread
            wide (30m) but shallow to catch rainwater. Cones need fire's heat to pop open! They grow 1m wider every 50 years.
            Some are 3,000 years old - alive when Rome was founded!
            """),
            [
                ("Why don't sequoias get taller than 95m?",
                """
                Gravity limits their water pumps! At 95m, tree veins struggle to lift water from roots to top needles. The
                tallest sequoia (Hyperion) is 115m but sick from water stress. Most stop growing up after 500 years, focusing on
                getting wider!
                """),
                ("How do fires help sequoia babies grow?",
                """
                Flames clear brush so seeds get sunlight! Heat opens cones to release 200,000 seeds. Ash fertilizes soil.
                Parent trees survive fires thanks to thick bark. Baby sequoias grow fast in cleared, sunny patches - fire is
                their nursery!
                """)
            ]
        ),
        (
            ("Tell me the story of how the Grand Canyon was carved",
            """
            The Colorado River started carving 6 million years ago! Water carries sand that sandpapers rocks. Winter freezes
            crack cliff edges. Summer rains cause mudslides. Each layer reveals Earth's history - 40+ rock types! The canyon
            grows 1cm wider yearly. At 446km long, it shows 2 billion years of geological stories!
            """),
            [
                ("Why are there different colored rock layers?",
                """
                Each color is a different ancient environment! Red layers = iron-rich deserts. Gray layers = deep ocean mud.
                White layers = volcanic ash. Green layers = swampy forests. The canyon is like Earth's history book with
                colored chapter pages!
                """),
                ("How do animals survive in the canyon's heat?",
                """
                Desert bighorn sheep drink morning dew from fur. Squirrels shade under cactus pads. Ravens steal hikers' water.
                Lizards dance on hot sand to keep feet cool. Scorpions glow under UV light to hunt at night - the canyon never
                sleeps!
                """)
            ]
        ),
        (
            ("How do arctic terns make their pole-to-pole migration?",
            """
            Arctic terns fly 71,000km yearly - longest animal migration! They chase endless summer between Arctic nesting
            grounds and Antarctic feeding zones. Using sun position and star maps, they zigzag to find best winds. A 30-year-old
            tern flies 2.4 million km - 3 trips to the moon and back!
            """),
            [
                ("How do terns sleep during 8-month flights?",
                """
                They nap while gliding! Terns shut half their brain at a time, like dolphins. One eye stays open to avoid
                obstacles. They catch 5-minute power naps during slow glides. At night, they float on ocean waves like feathered
                boats!
                """),
                ("Why don't terns get lost over featureless oceans?",
                """
                Built-in compasses! Terns sense Earth's magnetism through eye proteins. They memorize star patterns as chicks.
                Smells guide them near coastlines. Old terns teach young the route - it's a multi-generational GPS!
                """)
            ]
        )
      ]
      ''',
            4: '''
      [
        (
            ("Explain the Mariana Trench's formation and exploration",
            """
            The Mariana Trench is Earth's deepest point (11,034m) where two tectonic plates collide. Formed over 180 million years,
            it's deeper than Everest is tall. The Trieste submersible first reached Challenger Deep in 1960 with 2 crew. Pressure
            here equals 1,091 elephants standing on a soda can!
            """),
            [
                ("How did Trieste survive the trench's pressure?",
                """
                Trieste had 12cm thick steel walls shaped like a balloon. Its observation window was 15cm thick cone-shaped glass.
                Ballast tanks held gasoline (lighter than water). Dive took 5 hours down, 20 minutes exploring, 3 hours up.
                """),
                ("What unique life exists in total darkness?",
                """
                Ghostly white amphipods (shrimp-like) swarm thermal vents. Snailfish have gelatinous bodies resisting pressure.
                Giant single-celled xenophyophores make mud castles. All rely on 'marine snow' - dead plankton from above.
                """),
                ("Could modern subs improve exploration time?",
                """
                DSV Limiting Factor (2019) uses ceramic composites for 4-hour bottom time. It maps trenches with laser scanners.
                Robotic arms collect samples. Mothership tracks position via acoustic pingers. Still needs 2.5-hour descent.
                """),
                ("How does the trench shape global oceans?",
                """
                Trench absorbs Pacific Plate, recycling seawater into mantle. This regulates ocean chemistry over millennia.
                Cold seeps release methane feeding chemosynthetic life. Hadal zones may hold climate change clues.
                """)
            ]
        ),
        (
            ("Describe Roman aqueduct engineering secrets",
            """
            Romans built 11 aqueducts for 1M people. Used gravity flow with 0.004% slope precision. Channels had sedimentation tanks
            and inverted siphons. Arcades (above-ground arches) covered 80km of 420km total length. Some still work after 2,000 years!
            """),
            [
                ("How did they measure slopes without lasers?",
                """
                Used chorobates (5m water level tool) and groma (right-angle rods). Marked elevation changes with red paint poles
                every 20m. Slaves dug trial trenches first to test gradients.
                """),
                ("What's the Pont du Gard's special feature?",
                """
                This 3-tiered bridge has 52 arches spanning 275m. Upper channel narrows to increase water speed. Lower tiers
                widen to distribute weight. Built without mortar - stones cut to 5mm precision.
                """),
                ("How did inverted siphons work uphill?",
                """
                Lead pipes (15cm diameter) ran down valleys then up. Water pressure from descent pushed it upward. Stone pressure
                towers every 400m prevented pipe bursts. Required 30m minimum elevation difference.
                """),
                ("Why are Roman aqueducts still standing?",
                """
                Volcanic ash concrete gets stronger underwater. Limestone deposits (calcite) self-healed cracks. Strategic
                arcade designs distributed earthquake forces. Modern engineers still study their stress patterns.
                """)
            ]
        ),
        (
            ("How do tornadoes form and escalate?",
            """
            Tornadoes birth from supercell thunderstorms. Wind shear creates horizontal spinning tubes. Updrafts tilt them vertical.
            Funnel cloud descends when pressure drops 100hPa. EF5 tornadoes have 500km/h winds - faster than Formula 1 cars!
            """),
            [
                ("Why do some storms make tornadoes and others don't?",
                """
                Requires CAPE >2,500 J/kg (convective energy), wind shear >20m/s, and helicity >300 m²/s². Dryline boundaries
                in Tornado Alley mix hot/cold air perfectly. Only 20% of supercells spawn tornadoes.
                """),
                ("What's the 'dead man walking' tornado shape?",
                """
                Wedge tornadoes (1.6km wide) look like dark walls. Multiple vortices spin inside like drill bits. Debris balls
                glow red from power line sparks. The 2013 El Reno tornado reached 4.2km wide - widest ever.
                """),
                ("How do Doppler radars track tornado winds?",
                """
                Dual-polarization radar detects debris signatures. Velocity data shows rotation (couplet). Phased array radars
                update every 30 seconds. Mobile radars like DOW get within 1km of funnels.
                """),
                ("Can we stop tornadoes from forming?",
                """
                ️ Cloud seeding tried in Project Cirrus (1947). Silver iodide reduces hail but may intensify rotation. Modern
                focus is prediction - 13-minute average warning time. Underground shelters save 90%+ lives.
                """)
            ]
        ),
        (
            ("Explain Venus flytrap hunting mechanisms",
            """
            Venus flytraps have hinged leaves with trigger hairs. Two touches in 20 seconds slam shut. Digestive juices break
            down insects over 5-12 days. Red inner traps mimic flowers. Native only to 100km² in North Carolina.
            """),
            [
                ("How do trigger hairs work without nerves?",
                """
                Cells at hair base stretch when bent. Electrical signal (action potential) travels through aquaporin water
                channels. Second touch increases calcium ions to threshold - snap!
                """),
                ("Why don't traps close for raindrops?",
                """
                Raindrops lack nitrogen compounds. Sensors detect chitin (insect exoskeleton). Sugar secretions attract prey.
                False alarms cost energy - plant waits 24h before reopening.
                """),
                ("How do they avoid digesting themselves?",
                """
                Inner glands secrete digestive fluids only when touch sensors confirm prey. Waxy cuticle protects trap walls.
                pH drops to 2 during digestion - similar to stomach acid.
                """),
                ("Could they evolve to eat larger animals?",
                """
                Unlikely - energy cost exceeds gain. Current size (4cm traps) optimized for ants/beetles. Giant versions would
                need vascular systems absent in plants. But genetically modified traps could theoretically catch small frogs.
                """)
            ]
        ),
        (
            ("Describe Mars rover exploration challenges",
            """
            Mars rovers face -140°C nights, dust storms blocking sun, and terrain 5x worse than deserts. Perseverance (2021)
            has 43 sample tubes, laser spectrometers, and a helicopter scout. Signals take 5-20 minutes Earth-Mars travel time.
            """),
            [
                ("How do rovers survive Martian winters?",
                """
                Radioisotope heaters (plutonium-238) provide constant 110°C. Solar panels tilt toward sun. Electronics enter
            deep sleep - only clock and heaters active. Wheels lock to prevent drift.
                """),
                ("What's special about Ingenuity helicopter?",
                """
                Carbon fiber blades spin 2,400 RPM (5x Earth copters) to bite thin air. Lithium batteries charge via solar.
                Cameras map safe landing zones. Survived 25 flights despite design for 5. Proved aerial Mars exploration.
                """),
                ("How do lasers analyze rocks remotely?",
                """
            SuperCam laser zaps rocks (7m away) creating plasma. Spectrometer reads light signatures. Detects organic compounds
            and mineral types. Can clear dust spots with 30 laser blasts.
                """),
                ("Could future rovers support human missions?",
                """
                YES! MOXIE experiment makes oxygen from CO₂ (5g/hour). Rovers could prep landing sites and stockpile resources.
                Autonomous navigation avoids waiting for Earth commands. Dust removal systems needed for solar farms.
                """)
            ]
        )
      ]
      ''',
            6: '''
      [
        (
            ("How did Roman hypocaust systems heat buildings?",
            """
            Romans circulated hot air under floors using wood-fired furnaces. The hypocaust created raised floors (suspensurae)
            with brick pillars (pilae). Heat traveled through clay pipes in walls, warming bathhouses to 50°C. Wealthy villas had
            multi-room systems managed by slaves called fornacatores.
            """),
            [
                ("What materials prevented floor collapse?",
                """
                Layers of terracotta tiles (bipedales) spread heat evenly. Mortar mixed with volcanic ash (pozzolana) strengthened
                arches. The upper floor used marble chips in concrete for insulation. Regular lime coating prevented fire risks.
                """),
                ("How was temperature regulated?",
                """
                Bronze valves (regulae) controlled airflow. Slave-operated dampers adjusted furnace intensity. Bath progression
                (frigidarium to caldarium) naturally managed heat exposure. Window shutters timed solar gain.
                """),
                ("What maintenance challenges existed?",
                """
                Soot removal required monthly dismantling. Sulfur gases corroded bronze fittings. Mice nests in ducts caused
                uneven heating. Aqueduct-fed systems risked mineral deposits (calcare) blocking pipes.
                """),
                ("How did hypocausts influence Roman culture?",
                """
                Public baths became social hubs. Doctors prescribed heat therapies. Architects developed the testudo (heated
                niche) design. Fuel shortages led to deforestation laws (Lex Hordionia).
                """),
                ("Why did hypocaust use decline?",
                """
            Barbarian invasions disrupted fuel supplies. Christian asceticism discouraged luxury. Earthquakes damaged
            underground structures. Medieval reuse of materials for churches destroyed remaining systems.
                """),
                ("What modern systems derive from hypocausts?",
                """
            Radiant floor heating uses plastic PEX pipes instead of clay. Geothermal systems apply similar heat distribution.
            The Korean ondol and Islamic qanat heating preserve ancient principles. Museum preservation techniques
            stabilize original pilae.
                """)
            ]
        ),
        (
            ("Explain tardigrade cryptobiosis survival",
            """
            Tardigrades enter tun state by losing 97% body water. They produce trehalose sugar glass preserving cell structures.
            Special Dsup proteins protect DNA from radiation. Some survive -272°C to 150°C for decades.
            """),
            [
                ("How does tun formation work?",
                """
            Contractile proteins expel water in 30min. Organs shrink into compact shape. Metabolism drops to 0.01% normal.
            Antioxidants neutralize free radicals. Cell membranes become stacked lamellae.
                """),
                ("What's unique about Dsup proteins?",
                """
            Dsup binds DNA like protective cloud. Shields against X-rays and UV. Allows 1,000x more radiation than humans.
            May work by physical blocking rather than repair. Genetic engineers study it for astronaut protection.
                """),
                ("Can they survive space vacuum?",
                """
            Yes in 2007 ESA experiment. 68% revived after 10 days exposure. Survived solar UV by entering tun state. Eggs also
            survived. Proves panspermia possibility but not evidence.
                """),
                ("How do they revive from tun?",
                """
            Rehydration triggers metabolic restart. Trehalose dissolves first, repairing membranes. Mitochondria reactivate
            in phases. Full recovery takes hours. Some cells apoptose to remove damage.
                """),
                ("What ecosystems need tardigrades?",
                """
            Moss colonies depend on their nitrogen cycling. Lichen symbiosis requires their waste. Glacier melt ecosystems use
            revived populations. Some birds spread eggs through feathers.
                """),
                ("Could humans use cryptobiosis?",
                """
            Medical trials for organ preservation. Trehalose studied for blood cell storage. Dsup tested in radiation therapy.
            Space agriculture research for drought crops. Ethical debates on human suspended animation.
                """)
            ]
        ),
        (
            ("Describe carbon nanotube space elevators",
            """
            Theoretical 100,000km cable from equator to counterweight. Carbon nanotubes provide needed tensile strength.
            Climbers use laser power to ascend. Aims to reduce launch costs from $2000/kg to $100/kg.
            """),
            [
                ("Why carbon nanotubes?",
                """
            Their 63 GPa strength beats steel 100x. Thermal conductivity prevents laser damage. Flexibility handles atmospheric
            turbulence. Purity requirements need 99.9999% defect-free alignment.
                """),
                ("How to handle orbital debris?",
                """
            Self-healing sheathing with shape-memory alloys. Electrodynamic tethers repel small particles. Radar networks
            predict avoidance maneuvers. Emergency segmentation protocols prevent cascade failures.
                """),
                ("Anchor point challenges?",
                """
            Ocean platforms need hurricane resistance. Geostationary position requires active stabilization. Saltwater
            corrosion vs graphene coatings. Power beaming stations face lightning risks.
                """),
                ("Climber design specifics?",
                """
            Maglev tracks prevent friction. Photovoltaic cells convert 40% laser energy. Radiation shielding for crews.
            Split into 20-ton payload modules. Emergency parachutes for lower atmosphere failures.
                """),
                ("Economic impacts?",
                """
            Space solar farms become viable. Asteroid mining profitability increases. Orbital hotels accessible to tourists.
            Debris removal services emerge. Global treaty needed for cable ownership.
                """),
                ("Phase 2 developments?",
                """
            Lunar elevator using Kevlar-zylon blends. Mars elevator from Phobos. Orbital ring infrastructure. Nanotube
            production scales to megaton levels. Climber speeds reach 200km/h.
                """)
            ]
        ),
        (
            ("How do bacteria self-heal concrete?",
            """
            Bacillus pseudofirmus spores added to concrete mix. When cracks form, water activates bacteria. They consume
            calcium lactate producing limestone. Seals cracks up to 0.8mm wide. Extends structure life 20+ years.
            """),
            [
                ("Optimal spore concentration?",
                """
            10⁵ spores per gram cement. Higher concentrations weaken concrete. Encapsulated in clay pellets for protection.
            Dormant for 200 years until activation.
                """),
                ("Crack repair process duration?",
                """
            Initial sealing in 3 weeks. Full strength recovery in 6 months. Temperature dependent: 30°C ideal. Winter repairs
            need calcium formate accelerator. Maximum 5 repair cycles per structure.
                """),
                ("Material compatibility issues?",
                """
            Reduces compressive strength 15%. Not compatible with fly ash additives. Steel reinforcement needs extra epoxy
            coating. Testing required for seismic zones. pH must stay below 10.5.
                """),
                ("Environmental benefits?",
                """
            Cuts cement production CO2 by 30%. Eliminates toxic repair resins. Stormwater pH neutralization. Urban heat
            island reduction from lighter concrete. Noise pollution decrease by avoiding demolition.
                """),
                ("Monitoring techniques?",
                """
            Fluorescent dye reveals bacterial activity. Ultrasound measures crack depth. Thermal imaging shows repair progress.
            DNA sampling tracks spore viability. AI predicts next repair needs.
                """),
                ("Future architectural uses?",
                """
            Living bridges adapt to load changes. Self-sealing underground bunkers. Mars habitat construction. Underwater
            coral reef supports. Earthquake-resistant foundations with shape memory alloys.
                """)
            ]
        ),
        (
            ("Explain the mirror neuron system",
            """
            Found in F5 brain area, these neurons fire when performing or observing actions. Linked to empathy, language
            evolution, and social learning. Dysfunction implicated in autism. First discovered in macaques during 1990s Parma
            experiments.
            """),
            [
                ("Role in language development?",
                """
            Broca's area evolved from F5 homolog. Gesture recognition preceded speech. Baby mimetic skills depend on mirror
            responses. Dyslexia shows reduced activation during phoneme processing.
                """),
                ("Autism spectrum correlations?",
                """
            fMRI shows 30% lower mirror activity. Delayed imitation milestones. Reduced yawn contagion response. Therapy
            focuses on rhythm-based neuron activation. Genetic links to CNTNAP2 gene expression.
                """),
                ("Sports training applications?",
                """
            Mental rehearsal activates same pathways. VR simulations boost mirror responses. Expert vs novice neural patterns
            predict skill acquisition. Concussion recovery uses action observation therapy.
                """),
                ("Neuroplasticity factors?",
                """
            Meditation increases gray matter density. Stroke rehabilitation uses mirror therapy for paralysis. Dopamine
            levels affect system responsiveness. Sleep consolidates observational learning.
                """),
                ("Cross-species comparisons?",
                """
            Songbirds show similar vocal mimicry systems. Octopus observational learning suggests analogs. Dogs exhibit
            contagious yawning. Mirror tests prove self-awareness in great apes/dolphins. AI neural nets attempt replication.
                """),
                ("Ethical implications?",
                """
            Manipulation through media overexposure. Enhanced interrogation techniques. Military training desensitization.
            AI empathy programming risks. Neuroenhancement debates for education.
                """)
            ]
        )
      ]
      ''',
            8: '''
      [
        (
            ("How do quantum effects enhance photosynthesis in plants?",
            """
            Certain algae use quantum coherence to achieve 95% energy transfer efficiency in photosynthesis. Protein matrices in
            chloroplasts create vibrationally structured environments that maintain quantum superposition for 400 femtoseconds.
            """),
            [
                ("What experimental proof exists for quantum effects?",
                """
                2D electronic spectroscopy shows oscillating signals at 77K. Femtosecond lasers track exciton pathways. Green
                sulfur bacteria demonstrate this most clearly. The Fenna-Matthews-Olson complex acts as quantum wire.
                """),
                ("How do plants prevent quantum decoherence?",
                """
                Chromophore molecules are spaced 15-20Å apart. Vibrational modes couple to electronic transitions. Lipid
                membranes filter disruptive thermal noise. Evolutionary pressure optimized protein structures over 2B years.
                """),
                ("Can this be replicated artificially?",
                """
                MIT's 2025 quantum dots achieved 85% efficiency but required cryogenic temps. DNA scaffolds arrange chromophores.
                Challenges include scaling and oxygen sensitivity. Potential for ultra-efficient solar panels.
                """),
                ("Medical applications?",
                """
                Cancer drug delivery systems using targeted quantum coherence. Photosensitizers for photodynamic therapy.
                Neurodegenerative disease research on protein folding. Bio-inspired quantum sensors for early diagnosis.
                """),
                ("Environmental impacts?",
                """
                Could reduce solar farm land use by 60%. Algae farms might sequester CO2 more efficiently. Risks of engineered
                organisms escaping labs. Patent wars over biomimetic IP.
                """),
                ("Evolutionary advantages?",
                """
                Survived 3 Snowball Earth events. Enabled Cambrian explosion through oxygen surplus. Deep-sea species use
                low-light quantum tunneling. Symbiotic relationships with coral reefs.
                """),
                ("Quantum computing parallels?",
                """
            Topological qubits mimic vibrationally-assisted transport. Error correction resembles noise filtering in
            photosystems. Both use entanglement for information transfer. Biomaterials inspire room-temperature quantum devices.
                """),
                ("Ethical considerations?",
                """
                Gene editing algae for industrial use risks ecosystem collapse. Military potential for energy weapons.
                Nanoparticle pollution from degraded quantum materials. Access inequality for clean energy tech.
                """)
            ]
        ),
        (
            ("What makes aerogels the world's lightest solids?",
            """
            Silica aerogels are 99.8% air with density 3kg/m³. Created through supercritical drying preventing pore collapse.
            Nano-porous structure scatters blue light, giving translucent appearance. Thermal conductivity 0.015W/mK.
            """),
            [
                ("Manufacturing challenges?",
                """
                Precise sol-gel process takes 7 days. Requires 60°C ethanol baths. 0.1mm thickness limit without cracking.
                CO₂ supercritical drying costs $5000/kg. New ambient pressure methods cut costs 80%.
                """),
                ("Space applications?",
                """
            NASA's Stardust captured comet dust at 6km/s. Insulates Mars rovers (-140°C nights). Proposed for orbital debris
            shields. Future use in space habitat insulation.
                """),
                ("Medical breakthroughs?",
                """
                Drug-loaded aerogel implants release chemo over 6 months. Artificial cartilage with 90% water content.
                Hemostatic sponges stop bleeding in 15s. Experimental lung surfactant carriers.
                """),
                ("Environmental remediation?",
                """
                Absorbs oil spills at 40x own weight. Mercury capture from water. CO₂ sequestration matrices. Sound dampening in
                cities. Radioactive waste stabilization.
                """),
                ("Economic limitations?",
                """
            Graphene aerogels cost $300/g. Limited production scale (100kg/yr). Brittleness requires polymer reinforcement.
            Recycling methods not established. Fire risk without flame retardants.
                """),
                ("Future materials?",
                """
            Cellulose aerogels from waste paper. 3D-printed titanium aerogels for bone implants. Programmable thermal expansion
            variants. Self-healing versions with microcapsules.
                """),
                ("Historical development?",
                """
            Samuel Kistler's 1931 silica experiments. 1960s NASA funding for space suits. 1990s commercialization for
            window insulation. 2020s metamaterial integration.
                """),
                ("Cultural impacts?",
                """
            Art installations using glowing aerogels. Museum preservation of waterlogged artifacts. Luxury architecture
            translucent walls. Ethical debates on military insulation for drones.
                """)
            ]
        ),
        (
            ("How do Greenland sharks live 400+ years?",
            """
            Cold metabolism (0.5°C growth/yr). High TMAO counters urea toxicity. Cartilaginous skeleton reduces cancer risk.
            Constant deep-sea pressure stabilizes proteins. Sexual maturity at 150 years.
            """),
            [
                ("Anti-aging mechanisms?",
                """
            DNA repair enzymes work at 1°C. Telomerase expression in muscle tissue. Antioxidant-rich liver oils. Collagen
            cross-linking prevents tissue stiffening.
                """),
                ("Ecological role?",
                """
            Scavenge 70% of whale fall nutrients. Control giant squid populations. Distribute hydrothermal vent microbes.
            Teeth contain Arctic climate records.
                """),
                ("Medical research?",
                """
            TMAO studies for kidney disease. Antifreeze glycoproteins for organ storage. Cancer resistance gene isolation.
            Wound healing compounds from skin mucus.
                """),
                ("Conservation challenges?",
                """
            Bycatch kills 100/yr in trawls. Slow reproduction (10 pups/decade). Ocean warming reduces oxygen. Toxic heavy
            metal bioaccumulation.
                """),
                ("Biotech applications?",
                """
            Cryoprotectants for freeze-drying vaccines. Deep-sea pressure simulation chambers. Anti-inflammatory compounds
            from liver. Marine-derived collagen supplements.
                """),
                ("Cultural significance?",
                """
            Inuit legends of sea spirits. 19th century oil lamps used shark liver. Viking navigators followed shark
            migrations. Modern ecotourism regulations.
                """),
                ("Climate change impact?",
                """
            Melting ice exposes UV-damaged skin. Changing currents disrupt mating. Invasive species competition. Carbon
            dating of eye lenses tracks ocean acidification.
                """),
                ("Ethical debates?",
                """
            De-extinction research using DNA. Captivity stress studies. Traditional hunting rights vs conservation.
            Pharmaceutical exploitation concerns.
                """)
            ]
        ),
        (
            ("What enables octopus camouflage?",
            """
            Chromatophores with 25,000 color cells per cm². Radial muscles expand pigment sacs. Iridophores reflect light via
            100nm platelet stacks. Leucophores scatter all wavelengths. Neural control bypasses brain via arm ganglia.
            """),
            [
                ("Neurological control?",
                """
            Each sucker has 10,000 neurons. Decentralized processing allows arm autonomy. Optical receptors in skin detect
            surroundings. Neurotransmitters alter cell transparency.
                """),
                ("Material science inspiration?",
                """
            Adaptive color-changing fabrics. Military camouflage systems. Solar panel coatings. Anti-glare screens.
            Photonic computer chips.
                """),
                ("Evolutionary advantages?",
                """
            Avoids 80% predator attacks. Mimics 15+ species (lionfish, sea snakes). Flash displays startle enemies. UV
            patterns communicate secretly.
                """),
                ("Medical applications?",
                """
            Neural prosthesis research. Burn victim camouflage tattoos. Endoscopic imaging improvements. Synthetic chromatophore
            drug delivery.
                """),
                ("AI training models?",
                """
            Computer vision pattern recognition. Distributed neural networks. Robot skin prototypes. Swarm intelligence
            algorithms. Marine biology VR simulations.
                """),
                ("Aquaculture challenges?",
                """
            Stress reduces color response 40%. Tank reflections confuse skin. Nutritional needs for pigment production.
            Disease detection through dulling.
                """),
                ("Climate change impacts?",
                """
            Ocean acidification weakens skin cells. Coral loss reduces mimicry habitats. Warming seas accelerate metabolism.
            Plastic pollution causes false signals.
                """),
                ("Ethical considerations?",
                """
            Marine lab stress experiments. Aquarium light manipulation ethics. Gene editing for enhanced colors. Biomimetic
            patents limiting research access.
                """)
            ]
        ),
        (
            ("How do termites build 10m tall mounds?",
            """
            Macrotermes mold 5 tons of soil using saliva cement. Internal tunnels maintain 31°C via passive ventilation. Fungus
            gardens convert cellulose to nutrients. CO₂/O₂ exchange through porous walls.
            """),
            [
                ("Architectural principles?",
                """
            North-south orientation minimizes sun exposure. Spiral channels create convection currents. Moisture traps
            condense morning dew. Central chimney stabilizes airflow.
                """),
                ("Material composition?",
                """
            Saliva-bound soil 2x stronger than concrete. pH 8 prevents microbial growth. Magnetic particles align for
            navigation. Hydrophobic exterior repels rain.
                """),
                ("Energy efficiency?",
            """
            Zero external energy input. Solar chimney effect ventilates. Thermal mass stabilizes temps. Waste heat from
            fungus metabolism recycled.
                """),
                ("Human construction?",
                """
            Zimbabwe's Eastgate Centre mimics termite cooling. 3D-printed earth buildings. Passive solar designs.
            Mycelium-based insulation. Earthquake-resistant foundations.
                """),
                ("Ecological impacts?",
                """
            Mounds create microhabitats for 100+ species. Soil turnover prevents desertification. Methane production from
            digestion. Carbon sequestration in mound walls.
                """),
                ("Colony communication?",
                """
            Head-banging vibrations signal threats. Pheromone trails mark food sources. Trophallaxis shares gut bacteria.
            Royal jelly epigenetics control caste.
                """),
                ("Climate threats?",
            """
            Heavy rainfall collapses 20% of mounds. Heat waves dry out fungus gardens. Pesticides disrupt pheromone
            systems. Invasive ant species competition.
                """),
                ("Ethical research?",
                """
            Colony destruction for study. Gene drives to control invasive species. Traditional knowledge exploitation.
            Biomimetic patent monopolies.
                """)
            ]
        )
      ]
      '''
        }

        self.long_example_strs = {
            8: '''
            [
                (
                    ("How do Arctic terns navigate during 70,000km migrations?", 
                     "Arctic terns use multiple compass systems: magnetic field detection through eye proteins, star patterns memorized during fledging, and polarized light gradients. They calibrate these daily by tracking sunrise/sunset angles. Juveniles follow experienced adults to learn routes."),
                    [
                        ("Describe pitcher plant trapping mechanisms", 
                         "Pitcher plants create slippery surfaces with wax crystals and downward-pointing hairs. Rainwater dilutes digestive enzymes in the pitcher. They emit nectar guides visible in UV light to lure insects. Some species host symbiotic mosquito larvae to break down prey."),
                        ("How do they avoid self-digestion?", 
                         "Inner walls have waxy zones where enzymes can't pool. A collar-like peristome prevents rainwater overflow. Specialized cells secrete buffering agents to maintain optimal pH. Roots stay isolated from the digestive chamber."),
                        ("What adaptations prevent nutrient overload?", 
                         "Trap lids adjust opening based on prey size. Excess nutrients trigger temporary enzyme production shutdown. Symbiotic bacteria populations are regulated through antimicrobial secretions. Older pitchers transition to photosynthetic roles."),
                        ("How do tropical vs temperate species differ?", 
                         "Tropical pitchers use color gradients to target specific insects. Temperate species rely more on scent. High-altitude varieties have hairy insulation. Desert-adapted species use reflective lids to prevent evaporation."),
                        ("Impact on local ecosystems?", 
                         "Control mosquito populations. Provide microhabitats for 40+ specialist species. Enrich soil through controlled decomposition. Some frogs lay eggs in safe zones above digestive fluid."),
                        ("Conservation challenges?", 
                         "Poaching for medicinal myths. Habitat loss from peatland drainage. Invasive species disrupt symbiotic relationships. Climate change alters prey availability. Illegal plant trade reduces genetic diversity."),
                        ("Biomimetic applications?", 
                         "Self-cleaning surfaces mimic waxy textures. Robotic grippers inspired by directional hairs. Enzyme-regulated wastewater treatment systems. Architectural rainwater drainage designs based on pitcher shapes."),
                        ("How do Arctic terns adjust to changing magnetic fields?", 
                         "Arctic terns recalibrate navigation systems during stopovers by sampling magnetic field variations. Young birds update star maps through celestial drift observations. Adults shorten migration routes when encountering shifted prey zones.")
                    ]
                ),
                (
                    ("Why do baobab trees store water in swollen trunks?", 
                     "Baobabs absorb seasonal rains through spongy outer wood, storing up to 120,000 liters. The inner heartwood acts like a giant sponge, while fire-resistant bark prevents evaporation. Trees can lose 90% stored water during droughts without dying."),
                    [
                        ("How do electric eels generate 600-volt shocks?", 
                         "Three abdominal electricity-producing organs contain stacked electrocytes. Sodium/potassium ion pumps create charge differences. Nervous system signals synchronize 6,000 cells to discharge simultaneously. Fat layers insulate vital organs."),
                        ("What prevents them from electrocuting themselves?", 
                         "Current flows outward from tail tip. Insulating connective tissue surrounds vital organs. Brain floats in conductive gel that dissipates shocks. Charge distribution follows water conductivity paths away from body."),
                        ("How do they hunt with electricity?", 
                         "Low-voltage pulses locate hidden prey through electrolocation. High-voltage blitz causes involuntary muscle spasms. Curved body shape creates circular current fields. Juveniles practice on plant matter before live prey."),
                        ("Social behaviors observed?", 
                         "Groups coordinate shock attacks. Males guard nests with warning zaps. Electrocommunication signals convey size/status. Elders teach hunting strategies. Territorial displays involve synchronized discharges."),
                        ("Impact on aquatic ecosystems?", 
                         "Control invasive fish populations. Oxygenate stagnant waters. Distribute minerals through electrified paths. Create temporary electroplankton blooms. Dead eels fertilize riverbeds."),
                        ("Biomedical research applications?", 
                         "Pacemaker tech inspired by discharge control. Neural implants using electrocyte principles. Electric field cancer detection methods. Muscle stimulation therapies from spasm induction studies."),
                        ("Conservation status?", 
                         "Threatened by dam projects fragmenting habitats. Mercury contamination disrupts electrolocation. Overfishing for aquarium trade. Climate change alters freshwater conductivity. Protected in Amazon reserves."),
                        ("How do baobabs survive extreme droughts?", 
                         "Baobabs tap deep groundwater through 40m lateral roots. Stored water is rationed through vascular constriction. Leaves shed completely to prevent transpiration. Photosynthesis continues through green bark during dormancy.")
                    ]
                ),
                (
                    ("How do platypuses detect prey without sight?", 
                     "Platypus bills contain 40,000 electroreceptors and 60,000 mechanoreceptors. They hunt with eyes closed, creating 3D electrical maps of moving prey. Unique swimming motions maximize sensor coverage. Brain processes signals 10x faster than visual input."),
                    [
                        ("Why do alpine ibex climb vertical dams?", 
                         "Ibex hooves spread into rubbery suction cups. Rotating shoulder joints allow 180° limb movement. Mineral deposits on dam walls provide calcium supplements. Young learn climbing through observed trial/error over 3 years."),
                        ("How do dams support ibex populations?", 
                         "Concrete mimics natural cliff textures. Morning condensation provides water. Thermal mass retains heat. Predator access is limited. Winter ice prevents vegetation overgrowth on climbing surfaces."),
                        ("Unique social structures?", 
                         "Matriarchal herds rotate grazing/climbing duties. Males compete through vertical races. Elders teach kids safe paths. Separate winter/summer territories. Vocalizations echo off dam walls for communication."),
                        ("Health impacts of human structures?", 
                         "Iron reinforcements cause magnetic interference. Concrete dust affects digestion. Artificial lighting disrupts circadian rhythms. Road salt runoff creates mineral imbalances. Collision risks with cables."),
                        ("Conservation efforts?", 
                         "Special climbing corridors built. Mineral licks supplement nutrition. Nighttime light curfews. Anti-slip coatings tested. Population monitoring through drone thermal imaging."),
                        ("Biomimetic applications?", 
                         "Suction grippers for robotics. Shock-absorbing shoe soles. Dam inspection drones with ibex movement patterns. Earthquake-resistant building designs. Mountain rescue equipment improvements."),
                        ("Climate change effects?", 
                         "Altered concrete expansion/contraction. Increased lightning strikes on metal parts. Vegetation patterns shift. Extreme rains wash away mineral deposits. Heat waves reduce climbing activity windows."),
                        ("How do platypuses adapt to murky waters?", 
                         "Platypuses increase electroreception sampling rates in turbid conditions. They perform grid-search patterns with bills. Juveniles practice in sediment-stirred nurseries. Adults memorize productive hunting grounds during clearwater periods.")
                    ]
                ),
                (
                    ("Why do humpback whales create spiral bubble nets?", 
                     "Humpbacks exhale air in rising spirals to trap krill/fish. The bubble wall reflects their calls, stunning prey. They coordinate in groups, with some singing to herd prey while others blow bubbles. Net diameter matches prey swarm sizes (3-20m)."),
                    [
                        ("How do baobab trees store 120,000 liters of water?",
                         """
                         Spongy parenchyma tissue absorbs rainwater through surface roots. Fire-resistant bark (5cm thick) minimizes 
                         evaporation. Crisscrossing wood fibers expand trunk diameter 40% during wet seasons. Taproots reach 40m depth 
                         for groundwater access. Photosynthetic bark sustains trees during 9-month droughts.
                         """),
                        ("What enables 1000-year lifespans?",
                         """
                         Secondary metabolites prevent fungal decay. Modular vascular system isolates damage. Fire triggers protective 
                         cork cambium growth. DNA repair enzymes remain active in ancient specimens. Hollow trunks compartmentalize 
                         microbial ecosystems.
                         """),
                        ("Pollination strategies?",
                         """
                         Night-blooming flowers attract hawk moths with UV guides. 20cm nectar tubes match moth proboscis length. 
                         Rancid odor mimics mammal carrion. Pollen remains viable during 3-day flower lifespan. Bats disperse seeds 
                         through vitamin-C-rich fruit.
                         """),
                        ("How do elephants interact with baobabs?",
                         """
                         Bulls rub tusks on bark to calcium-harden ivory. Herds strip leaves during dry seasons. Seed distribution 
                         through dung boosts germination 300%. Tusks carve water reservoirs in trunks. Calves learn migration routes 
                         via baobab landmarks.
                         """),
                        ("Climate change adaptations?",
                         """
                         Flowering shifted 6 weeks earlier. Trunk fissures trap moisture from fog. Some populations developed 
                         deciduous traits. Mycorrhizal networks share water between trees. Carbon isotope ratios show increased 
                         water-use efficiency.
                         """),
                        ("Cultural significance?",
                         """
                         African oral histories call them 'Upside-Down Trees'. Malagasy use hollow trunks for village councils. 
                         Australian Aboriginal calendars track flowering times. Colonial explorers used them as navigation beacons. 
                         Modern Botswana banknotes feature baobabs.
                         """),
                        ("Conservation innovations?",
                         """
                         Drone pollination supplements hawk moth declines. Biochar injections prevent hollow trunk collapse. 
                         Elephant-resistant fencing uses fermented bark scent. Community-led seed banking preserves genetic diversity. 
                         Satellite monitoring tracks climate responses.
                         """),
                        ("How do humpbacks adapt bubble nets to changing prey?", 
                         """
                        Humpbacks adjust bubble net sizes based on krill swarm density. They dive deeper during plankton blooms. Young whales learn spiral patterns through vocal mimicry. Groups coordinate net angles using body slaps.
                        """)
                    ]
                )
            ]
            ''',
            12: '''
            [
                (
                    ("How do Arctic foxes adapt to winter camouflage and hunting?",
                     """
                     Arctic foxes transition coats over 6 weeks using melanocyte-stimulating hormones. Underfur density triples with 2cm-long 
                     hollow guard hairs for insulation. Paw pads shrink 30% to reduce heat loss. They develop ultraviolet vision to spot prey 
                     against snow, while specialized kidney functions allow them to hydrate from frozen meat.
                     """),
                    [
                        ("Describe pitcher plant digestive adaptations",
                         """
                         Pitchers produce nepenthesin enzymes that break down proteins at pH 2.5. Waxy inner walls prevent insect escapes. 
                         Symbiotic mosquito larvae churn fluid to accelerate decomposition. Rainwater dilution triggers enzyme activation. 
                         Red-veined patterns guide prey to digestive zones.
                         """),
                        ("How do tropical vs temperate species differ?",
                         """
                         Tropical varieties have elongated pitchers (50cm) for arboreal prey. Temperate species use ground-level traps with 
                         antimicrobial nectar. Highland types maintain fluid warmth through hairy insulation. Desert-adapted species 
                         conserve water through sealed lids during droughts.
                         """),
                        ("What prevents self-digestion?",
                         """
                         Collar-like peristome channels fluid away from stem. Lignin-rich inner walls resist enzymes. Root systems secrete 
                         alkaline buffers. Glandular zones isolate digestive activity. Older pitchers develop non-sticky zones as they 
                         transition to photosynthetic roles.
                         """),
                        ("How do they attract specific prey?",
                         """
                         Ultraviolet patterns mimic insect mating signals. Volatile amines replicate rotting meat scents. Nectar contains 
                         addictive alkaloids. Some species emit ultrasonic clicks that confuse bat navigation. Tropical varieties use 
                         reflective lids to attract moths.
                         """),
                        ("Impact on local ecosystems?",
                         """
                         Control mosquito populations by trapping adults. Provide microhabitats for 40+ specialist species. Enrich soil 
                         through controlled nutrient release. Serve as emergency water sources for primates during droughts. Their DNA 
                         shows coevolution with specific ant species.
                         """),
                        ("Climate change adaptations observed?",
                         """
                         Alpine species develop thicker wax layers. Tropical varieties increase nectar production during irregular rains. 
                         Some hybrids show expanded temperature tolerance. Pollination timing shifted 2 weeks earlier in temperate zones. 
                         Desert species evolved deeper root systems.
                         """),
                        ("Biomimetic research applications?",
                         """
                         Self-cleaning coatings mimic waxy surfaces. Robotic grippers use directional hair designs. Enzyme-based wastewater 
                         treatment systems. Architectural rainwater collection inspired by lid mechanics. Air purification filters 
                         replicating volatile compound absorption.
                         """),
                        ("Conservation challenges?",
                         """
                         Poaching for traditional medicine increased 300% since 2015. Invasive ants disrupt symbiotic relationships. 
                         Atmospheric pollution alters scent molecule effectiveness. Illegal hybridization threatens genetic diversity. 
                         Ecotourism trampling damages microhabitats.
                         """),
                        ("How do nurseries cultivate them?",
                         """
                         Tissue culture clones maintain genetic purity. Artificial diets use gelatin-peptone mixes. LED arrays provide 
                         species-specific light spectra. Humidity domes prevent premature enzyme production. Ant colonies introduced 
                         during juvenile growth phases.
                         """),
                        ("Historical cultural significance?",
                         """
                         Borneo tribes used pitchers as water containers. Victorian collectors sparked 'Pitcher Mania'. Ayurvedic 
                         medicine employed extracts for digestion. Aboriginal dreamtime stories feature them as earth spirits. Modern 
                         Singaporean currency depicts local species.
                         """),
                        ("Future research directions?",
                         """
                         Genetic sequencing of digestive enzyme evolution. Biomimetic pesticide delivery systems. Climate-resilient hybrid 
                         development. Antimicrobial compound extraction for medicine. Microhabitat creation in urban pollution hotspots.
                         """),
                        ("How are Arctic foxes surviving warmer winters?",
                         """
                         Arctic foxes now extend coastal scavenging into ice-free months. Their coats transition 3 weeks later than historic 
                         averages. Some populations developed all-season gray morphs. Pups learn to hunt seabird colonies as traditional 
                         prey patterns shift northward.
                         """)
                    ]
                ),
                (
                    ("Why do narwhal tusks have 10 million nerve endings?",
                     """
                     The spiral tusk detects salinity (5000ppm resolution), temperature (0.1°C accuracy), and pressure changes. Nerve 
                     clusters map Arctic currents through vibration analysis. Tusks help locate breathing holes under ice and 
                     stun prey with targeted pressure waves. Growth rings reveal 50-year climate records.
                     """),
                    [
                        ("How do baobab trees store 120,000 liters of water?",
                         """
                         Spongy parenchyma tissue absorbs rainwater through surface roots. Fire-resistant bark (5cm thick) minimizes 
                         evaporation. Crisscrossing wood fibers expand trunk diameter 40% during wet seasons. Taproots reach 40m depth 
                         for groundwater access. Photosynthetic bark sustains trees during 9-month droughts.
                         """),
                        ("What enables 1000-year lifespans?",
                         """
                         Secondary metabolites prevent fungal decay. Modular vascular system isolates damage. Fire triggers protective 
                         cork cambium growth. DNA repair enzymes remain active in ancient specimens. Hollow trunks compartmentalize 
                         microbial ecosystems.
                         """),
                        ("Pollination strategies?",
                         """
                         Night-blooming flowers attract hawk moths with UV guides. 20cm nectar tubes match moth proboscis length. 
                         Rancid odor mimics mammal carrion. Pollen remains viable during 3-day flower lifespan. Bats disperse seeds 
                         through vitamin-C-rich fruit.
                         """),
                        ("How do elephants interact with baobabs?",
                         """
                         Bulls rub tusks on bark to calcium-harden ivory. Herds strip leaves during dry seasons. Seed distribution 
                         through dung boosts germination 300%. Tusks carve water reservoirs in trunks. Calves learn migration routes 
                         via baobab landmarks.
                         """),
                        ("Climate change adaptations?",
                         """
                         Flowering shifted 6 weeks earlier. Trunk fissures trap moisture from fog. Some populations developed 
                         deciduous traits. Mycorrhizal networks share water between trees. Carbon isotope ratios show increased 
                         water-use efficiency.
                         """),
                        ("Cultural significance?",
                         """
                         African oral histories call them 'Upside-Down Trees'. Malagasy use hollow trunks for village councils. 
                         Australian Aboriginal calendars track flowering times. Colonial explorers used them as navigation beacons. 
                         Modern Botswana banknotes feature baobabs.
                         """),
                        ("Conservation innovations?",
                         """
                         Drone pollination supplements hawk moth declines. Biochar injections prevent hollow trunk collapse. 
                         Elephant-resistant fencing uses fermented bark scent. Community-led seed banking preserves genetic diversity. 
                         Satellite monitoring tracks climate responses.
                         """),
                        ("Biomimetic applications?",
                         """
                         Water storage tanks mimic spongy wood structure. Fire-resistant materials based on bark chemistry. 
                         Architectural designs use modular vascular concepts. Drought-warning systems modeled on leaf drop patterns. 
                         Carbon capture inspired by trunk expansion.
                         """),
                        ("Medical research connections?",
                         """
                         Bark extracts show anti-malarial properties. Fruit powder reduces childhood malnutrition. Leaf compounds 
                         inhibit HIV replication. Seed oil accelerates burn healing. Pollen studies improve allergy vaccines.
                         """),
                        ("How do baobabs support desert ecosystems?",
                         """
                         Hollow trunks host 47 vertebrate species. Flowers feed 30+ pollinator species. Seeds survive elephant digestion. 
                         Bark shelters temperature-sensitive lichens. Morning dew collection sustains insect colonies.
                         """),
                        ("What threats are emerging?",
                         """
                         Invasive beetles tunnel through bark. Artificial lighting disrupts moth pollination. Groundwater extraction 
                         lowers water tables. Climate models predict 50% habitat loss by 2100. Traditional harvest practices decline.
                         """),
                        ("How are narwhals adapting to Arctic changes?",
                         """
                         Narwhals now dive 1500m for cold-water prey as surface temps rise. Tusks detect new salinity fronts from melted 
                         glaciers. Calves learn faster ice navigation through extended maternal care. Some pods shifted wintering grounds 
                         400km northward over 20 years.
                         """)
                    ]
                ),
                (
                    ("How do octopuses solve complex puzzles underwater?",
                     """
                     Octopuses use 500 million neurons split between brain and arms. Each sucker contains chemo-tactile sensors 
                     recognizing 100+ textures. They perform observational learning, tool use (coconut shelters), and future planning. 
                     Tests show problem-solving comparable to 5-year-old humans.
                     """),
                    [
                        ("Why do migrating monarchs cluster in oyamel firs?",
                         """
                         Fir canopies maintain 2-5°C microclimates. Resinous sap deters predators. Needle density breaks wind currents. 
                         Sunlight filters through in UV-rich patterns guiding cluster positions. Bark grooves trap heat during freezing 
                         nights.
                         """),
                        ("How do they navigate 4000km routes?",
                         """
                         Time-compensated sun compass using circadian rhythms. Magnetic field detection through cryptochrome proteins. 
                         Ultraviolet polarization patterns map directions. Valley landmarks trigger course corrections. Older 
                         butterflies lead fall migrations.
                         """),
                        ("What threatens overwintering sites?",
                         """
                         Illegal logging removed 45% of oyamel forests since 2000. Climate change causes early budburst mismatching 
                         arrivals. Pesticides reduce lifespan 60%. Tourism trampling damages microclimates. Invasive wasps parasitize 
                        30% of larvae.
                         """),
                        ("Conservation success stories?",
                         """
                         Community patrols reduced logging 80% in core zones. Milkweed corridors along highways expanded breeding. 
                         Citizen science tags tracked migration shifts. Cold storage preserves genetics. School programs promote 
                         butterfly gardening.
                         """),
                        ("How do they survive freezing temps?",
                         """
                         Antifreeze glycolipids in hemolymph. Cluster shivering maintains 10°C core. Wing scales trap insulating air 
                         pockets. Reduced metabolism survives 2 months without food. UV-reflective scales prevent ice nucleation.
                         """),
                        ("Biomimetic research applications?",
                         """
                         Thin-film solar cells mimic wing scales. Lightweight insulation inspired by cluster behavior. Flight algorithms 
                         for microdrones. Chemical sensors based on antennae receptors. Drug delivery using proboscis tube mechanics.
                         """),
                        ("Cultural significance?",
                         """
                         Aztec warriors believed monarchs carried souls. Day of the Dead celebrations coincide with migrations. 
                         Canadian/Mexican joint conservation stamps. Children's books feature migration as perseverance symbol. 
                         Biotech companies use monarch DNA in patents.
                         """),
                        ("How do larvae detect toxic milkweed?",
                         """
                         Chemoreceptors on antennae and feet detect cardenolides. Larvae balance toxin intake vs growth rate. 
                         Selective feeding avoids oldest leaves. Gut microbes neutralize 40% of toxins. Cannibalism removes 
                         poisoned individuals.
                         """),
                        ("Climate change adaptations observed?",
                         """
                         Spring migration starts 15 days earlier. Some populations use alternative host plants. Diapause periods 
                         shortened by 3 weeks. Wing sizes increased 5% for longer flights. Hybridization with southern relatives 
                         introduced heat tolerance.
                         """),
                        ("What mysteries remain unsolved?",
                         """
                         How navigational maps are genetically encoded. Why western populations don't migrate. Role of microbial 
                         communities in longevity. Exact magnetic field detection mechanism. Reasons for cyclical population 
                         fluctuations.
                         """),
                        ("Future conservation strategies?",
                         """
                         AI-powered deforestation monitoring. Genetically enhanced milkweed nutrition. International pesticide 
                         treaties. Climate-controlled overwintering greenhouses. Satellite tracking of micro-migration paths.
                         """),
                        ("How do octopuses' problem-solving skills aid survival?",
                         """
                         Octopuses now open shellfish faster as ocean acidification weakens shells. They repurpose plastic waste as 
                         camouflage tools. Some populations learned to avoid fishing traps through social learning. Arm regeneration 
                         speeds increased 25% in polluted zones.
                         """)
                    ]
                ),
                (
                    ("Why do hummingbirds enter torpor during cold nights?",
                     """
                     Torpor reduces metabolism 95%, dropping heart rate from 1200 to 50bpm. Body temp matches surroundings (5-21°C). 
                     They awaken via shivering thermogenesis powered by pectoral muscles. This conserves energy when nectar is scarce, 
                     surviving 8-hour freezes that would otherwise be fatal.
                     """),
                    [
                        ("How do leafcutter ants farm fungus?",
                         """
                         Colonies collect leaf fragments to grow Leucoagaricus gardens. Worker castes chew leaves into pulp, adding fecal 
                         enzymes. Soldier ants weed competing molds. The fungus converts cellulose into edible gongylidia. Queens carry 
                         starter cultures during nuptial flights.
                         """),
                        ("What maintains garden health?",
                         """
                         Antibiotic-producing bacteria on ant cuticles suppress pathogens. Waste chambers isolate contaminated material. 
                         Humidity sensors trigger ventilation digging. Older workers remove infected fungus. Pheromone markers designate 
                         compost zones.
                         """),
                        ("How do they coordinate harvesting?",
                         """
                         Pheromone trails guide 100m foraging paths. Scouts assess leaf quality through taste receptors. Load size adjusts 
                         based on colony needs. Solar navigation maintains straight paths. Rain triggers emergency retrieval protocols.
                         """),
                        ("What predators threaten colonies?",
                         """
                         Parasitic phorid flies lay eggs in ant heads. Army ant raids trigger evacuation plans. Fungal pathogens 
                         (Escovopsis) mimic pheromones. Anteaters destroy nests. Droughts collapse underground humidity controls.
                         """),
                        ("Biomedical research applications?",
                         """
                         Antibiotics from mutualistic bacteria treat resistant infections. Swarm intelligence algorithms optimize 
                         supply chains. Waste management systems inspire recycling tech. Fungal enzymes improve biofuel production. 
                         Pheromone studies advance pest control.
                         """),
                        ("Climate change impacts observed?",
                         """
                         Foraging distances increased 30% in dry zones. Night activity rose 50% to avoid heat. Fungal gardens require 
                         more frequent replacement. Queen fertility dropped 15%. Some species shifted to novel leaf sources.
                         """),
                        ("How do larvae develop caste roles?",
                         """
                         Nutritional programming determines worker/soldier/queen paths. Royal jelly activates ovary genes. Soldier 
                         larvae receive protein-rich diets. Temperature fluctuations influence size differentiation. Pheromone 
                         exposure during pupation fixes social behaviors.
                         """),
                        ("Evolutionary advantages?",
                         """
                         Farming allowed niche expansion into barren areas. Symbiosis outcompetes solitary species. Caste system 
                         enables complex labor division. Fungal enzymes digest toxic leaves. Collective intelligence solves 
                         resource challenges.
                         """),
                        ("What mysteries remain?",
                         """
                         How starter fungus survives in new queens. Precise pathogen detection methods. Evolutionary origin of 
                         caste determination. Role of vibrational communication. Reasons for garden color variations.
                         """),
                        ("Conservation importance?",
                         """
                         Soil aeration improves rainforest regeneration. Their trails shape understory plant diversity. 
                         Antibiotic compounds have undiscovered medical potential. Biomass exceeds vertebrates in some ecosystems. 
                         Climate change indicator species.
                         """),
                        ("Future research directions?",
                         """
                         Genetic modification of fungal crops. AI models of swarm decision-making. Microbial community transplants 
                         between colonies. Nanostructure studies of leaf-cutter jaws. Space station agricultural applications.
                         """),
                        ("How do hummingbirds optimize torpor use?",
                         """
                         Hummingbirds now enter torpor 3x more frequently during erratic blooms. Some species reduce nighttime 
                         temps to 3°C for deeper energy savings. Urban populations use streetlight heat to shorten torpor periods. 
                         High-altitude variants evolved faster thermogenesis rates.
                         """)
                    ]
                )
            ]
            ''',
            16: '''
            [
                (
                    ("How do Arctic foxes survive -50°C winters through metabolic adaptations?",
                     """
                     Arctic foxes reduce basal metabolic rate by 35% using thyroid hormone regulation. Subcutaneous fat layers insulate core organs, 
                     while countercurrent heat exchangers in legs prevent frostbite. They enter short-term torpor (4-6hrs) during blizzards, 
                     surviving on cached prey. Kidney functions concentrate urea to minimize water loss from breathing dry air.
                     """),
                    [
                        ("Describe strangler fig ecosystem engineering",
                         """
                         Strangler figs germinate in tree canopies, sending aerial roots downward. Roots fuse into lattice frameworks that 
                         eventually engulf hosts. Hollow trunks create microhabitats for 87+ species. Fig wasp symbiosis ensures pollination 
                         during synchronous fruiting.
                         """),
                        ("How do root systems stabilize host trees?",
                         """
                         Buttress roots redistribute weight during storms. Hydraulic lift shares groundwater with hosts. Chemical signals
                         suppress host decay fungi. Root exudates improve soil mycorrhizal networks. Some figs abort growth if hosts weaken.
                         """),
                        ("What drives synchronous fruiting?",
                         """
                         Photoperiod sensors in leaf tips trigger flowering. Ethylene gas coordinates regional fruiting. Mass fruiting
                         overwhelms seed predators. Temperature thresholds (25°C±2) activate enzyme production. Drought years delay cycles
                         by 6-8 months.
                         """),
                        ("How do vertebrates depend on figs?",
                         """
                         Fruit bats navigate via fig pheromone trails. Hornbills time nesting with fruit abundance. Orangutans get 80%
                         wet season calories from figs. Fish species evolved to eat flood-dispersed seeds. Extinct megafauna were key
                         dispersers.
                         """),
                        ("Climate change adaptations observed?",
                         """
                         Flowering advanced 2.3 days/decade. Some species developed drought-tolerant root caps. Highland figs shifted
                         fruiting altitudes. Hybridization created heat-resistant varieties. Pollinator wasps evolved faster life cycles.
                         """),
                        ("Biomedical applications?",
                         """
                         Latex contains tumor-inhibiting ficins. Root extracts show antimalarial quinine analogs. Bark compounds
                         stabilize insulin. Fig wasp venom studies inform painkillers. Hollow trunks inspire artificial organ scaffolds.
                         """),
                        ("Conservation challenges?",
                         """
                         Fragmented forests disrupt pollination. Invasive ants protect non-native pests. Climate mismatch separates
                         figs/wasps. Logging removes host trees. Traditional propagation knowledge disappearing.
                         """),
                        ("How do epiphytic communities form?",
                         """
                         Accumulated humus in root crevices supports orchids/ferns. Water-filled leaf axils breed tree frogs.
                         Beetles carve ventilation tunnels. Bioluminescent fungi light cavities. Some figs host ant colonies
                         for defense.
                         """),
                        ("Historical human uses?",
                         """
                         Mayan cities built around sacred figs. Bark cloth production sustained Pacific cultures. Traditional
                         medicine treated 40+ ailments. Living bridges last 500+ years. Colonial ships used fig latex as
                         caulk.
                         """),
                        ("Biomimetic innovations?",
                         """
                         Self-repairing materials mimic root fusion. Earthquake-resistant foundations copy lattice designs.
                         Fog-harvesting surfaces inspired by leaf textures. Hydraulic architecture modeled on root pressure
                         systems.
                         """),
                        ("Soil impact mechanisms?",
                         """
                         Root exudates lower soil pH for nutrient absorption. Calcium oxalate crystals buffer aluminum toxicity.
                         Termite clay deposition improves water retention. Decaying host wood releases slow-fertilizing nutrients.
                         Leaf litter suppresses competing plants.
                         """),
                        ("Pollination precision?",
                         """
                         Wasp antennae detect volatile terpenes. Flowers open in 6hr windows matching wasp activity. Heat-producing
                         florets maintain 32°C for wasp metabolism. Stigma receptivity peaks when pollen tubes are primed.
                         Chemical mimicry prevents interspecies hybridization.
                         """),
                        ("Seed dispersal efficiency?",
                         """
                         Fruit bats excrete 90% seeds within 1km. Fish gut acids scarify seeds. Floods distribute 800,000 seeds/ha.
                         Elephants transported seeds 60km pre-extinction. Ants cache seeds in optimal germination sites.
                         """),
                        ("How do young figs avoid self-strangulation?",
                         """
                         Delayed root lignification allows host growth. Flexible root collars accommodate trunk expansion.
                         Photosynthetic roots supplement energy. Chemical signals coordinate growth pauses. Some species
                         maintain permanent aerial roots.
                         """),
                        ("Future research priorities?",
                         """
                         Gene-editing for faster maturation. Artificial pollination techniques. Mycorrhizal inoculation
                         protocols. Carbon sequestration quantification. Climate-resilient hybrid development.
                         """),
                        ("How are Arctic foxes adapting to thawing permafrost?",
                         """
                         Arctic foxes now dig deeper dens in unstable ground. Their winter torpor periods shortened by 2hrs
                         to monitor collapsing snow tunnels. Increased scavenging of marine mammals compensates for declining
                         lemming cycles. Paw pads evolved rougher textures for ice-free terrain.
                         """)
                    ]
                ),
                (
                    ("Why do narwhal tusks detect climate change signatures?",
                     """
                     Narwhal tusks accumulate oceanic data in annual growth layers. Stable isotopes (δ¹⁸O, δ¹³C) reveal 
                     temperature/food web changes. Trace metals track pollution timelines. Collagen proteins indicate 
                     metabolic stress. Some tusks contain 50-year environmental records with monthly resolution.
                     """),
                    [
                        ("How do corpse flowers generate metabolic heat?",
                         """
                         Titan arum inflorescences reach 32°C via cyanide-resistant respiration. Mitochondrial thermogenin 
                         proteins uncouple ATP production. Heat volatilizes cadaverine/scatole to attract pollinators. 
                         Temperature gradients guide carrion beetles to pollen. Energy equivalent to 80,000 human cells 
                         per gram.
                         """),
                        ("Pollination strategy details?",
                         """
                         Female-stage flowers mimic fresh carcass chemistry. Beetles get trapped overnight for pollen loading.
                         Male-stage pollen mimics maggot movement through thermonastic stamen vibrations. UV patterns guide
                         exit paths. 98% pollination failure in nature.
                         """),
                        ("How do seedlings survive nutrient-poor soils?",
                         """
                         Mycorrhizal networks connect to 30+ tree species. Cotyledons photosynthesize for 18 months.
                         Carnivorous root hairs trap nematodes. Allelopathic chemicals suppress competitors.
                         Giant tubers (50kg) store 10-year energy reserves.
                         """),
                        ("Climate change impacts?",
                         """
                         Flowering intervals shortened from 7 to 4 years. Heatwaves cause 40% abortion rates.
                         Pollinator beetle ranges shifted 300km north. Reduced fog increases seedling mortality.
                         Genetic diversity dropped 70% in fragmented populations.
                         """),
                        ("Biotechnological applications?",
                         """
                         Thermogenic proteins improve vaccine cold chains. Volatile compound detectors inspired
                         by scent mechanisms. Root networks model underground internet systems. Flower structure
                         informs passive cooling architecture.
                         """),
                        ("Conservation innovations?",
                         """
                         DNA banks preserve genetic diversity. Hand pollination increases seed set 90%.
                         Fog capture systems hydrate seedlings. Thermal imaging tracks flowering events.
                         Community patrols deter illegal bulb harvesting.
                         """),
                        ("Historical cultural roles?",
                         """
                         Sumatran shamans used blooms in death rituals. Victorian collectors caused 'Titan mania'.
                         Japanese animators featured them as monster prototypes. Modern botanical gardens use
                         blooms to fund conservation.
                         """),
                        ("Seed dispersal mechanisms?",
                         """
                         Hornbills swallow fruits whole. Elephants crushed 99% seeds pre-extinction.
                         Floods distribute remaining seeds. Ants cache seeds but eat elaiosomes.
                         Gravity dispersal now predominant without megafauna.
                         """),
                        ("How do nurseries cultivate them?",
                         """
                         Tissue culture clones mature 3x faster. Artificial tubers provide 6-year energy.
                         Pollen stored in liquid nitrogen. Growth chambers mimic Bornean humidity.
                         AI monitors millimeter-scale growth changes.
                         """),
                        ("Metabolic research insights?",
                         """
                         Alternative oxidase pathways inform cancer studies. Protein folding during heat
                         production aids vaccine research. Lipid mobilization models improve energy storage tech.
                         Floral thermoregulation inspires smart materials.
                         """),
                        ("Ecological relationships at risk?",
                         """
                         ‍♀️ Obligate pollinator beetles face extinction. Mycorrhizal fungi networks
                         disrupted by logging. Seed dispersal void left by lost megafauna. Climate-driven
                         range mismatches with host trees. Invasive ants prey on crucial pollinators.
                         """),
                        ("Genetic uniqueness?",
                         """
                         Genome size varies 300% between populations. Horizontal gene transfer from
                         symbiotic fungi detected. Retrotransposons drive rapid morphological changes.
                         Epigenetic methylation patterns track habitat degradation.
                         """),
                        ("Public engagement strategies?",
                         """
                         Live bloom webcams reach 10M viewers. Adopt-a-Tuber programs fund conservation.
                         Citizen science tracks flowering smells. School curricula feature life cycle
                         mysteries. VR experiences simulate pollination journeys.
                         """),
                        ("How do temperature changes affect reproduction?",
                         """
                         Nighttime warming reduces scent volatility. Beetle attraction drops 60% above
                         35°C. Pollen viability halves per 2°C increase. Flowering peaks now mismatch
                         beetle emergences. Thermogenic failure causes 80% aborted blooms.
                         """),
                        ("Future research directions?",
                         """
                         Engineered pollinators via gut microbiome transplants. Synthetic pheromone
                         lures. Cryopreservation of giant tubers. Satellite tracking of seed dispersal.
                         Quantum dot labeling of floral nutrients.
                         """),
                        ("How are narwhals tracking Arctic ecosystem shifts?",
                         """
                         Narwhal tusks now show 40% higher mercury levels from thawing permafrost. Growth layers
                         indicate novel prey species moving north. Tusk nerve endings detect salinity changes
                         from glacial melt. Pods restructure migration routes around ice-free corridors.
                         """)
                    ]
                ),
                (
                    ("How do octopuses use RNA editing for environmental adaptation?",
                     """
                     Octopuses edit 60% neural RNA to rapidly adjust protein structures. This allows 
                     temperature/pH tolerance changes within hours. Edited kinesin proteins repair 
                     cold-damaged neurons. Voltage-gated channels adapt to varying salinity. Memory 
                     formation links RNA editing to learned behaviors.
                     """),
                    [
                        ("Describe rafflesia's parasitic lifecycle",
                         """
                         Rafflesia seeds infect Tetrastigma vine cambium. Fungal-like filaments consume 
                         host 5 years before blooming. Flowers mimic rotting meat via thermogenesis 
                         (36°C) and dimethyl disulfide. No roots/leaves - 100% host-dependent. 
                         Genome reduced to 44Mb through gene loss.
                         """),
                        ("How do they avoid host rejection?",
                         """
                         Suppress host defenses with cytokinin mimics. Cell wall softening enzymes
                         enable nutrient transfer. MicroRNA silences host immune genes. Shared
                         xylem/phloem connections avoid detection. Chemical camouflage matches host
                         VOC profiles.
                         """),
                        ("Pollination challenges?",
                         """
                         Flowers bloom 5 days annually. Female/male flower separation prevents
                         selfing. Carrion flies get trapped in slippery chambers. Only 0.3% of
                         blooms get pollinated. Seeds require elephant foot transport.
                         """),
                        ("Conservation status?",
                         """
                         Critically endangered - 20/32 species extinct since 1980. Habitat loss
                         removed 90% host vines. Climate change disrupts synchronized blooming.
                         Overharvesting for traditional medicine. No successful ex-situ cultivation.
                         """),
                        ("Biomedical potential?",
                         """
                         Immunosuppressive compounds prevent transplant rejection. Tumor-like growth
                         mechanisms inform cancer research. Heat-production proteins studied for
                         hypothermia treatment. Ultra-rare genetic code aids viral vector design.
                         """),
                        ("How do locals utilize rafflesia?",
                         """
                         Buds treat postpartum pain. Fibers bind ritual objects. Bloom events mark
                         agricultural cycles. Ecotourism provides alternative income. Some cultures
                         associate it with evil spirits.
                         """),
                        ("Evolutionary origins?",
                         """
                         Diverged from poppy relatives 46MYA. Lost photosynthesis genes through
                         horizontal transfer. Genome miniaturization parallels deep-sea parasites.
                         Flower gigantism evolved 3 times independently. Mitochondrial genes
                         transferred to host nucleus.
                         """),
                        ("Seed survival strategies?",
                         """
                         Seeds mimic host fruits to attract dispersers. Thick cuticle survives
                         digestive acids. Chemical triggers delay germination until host contact.
                         Mycorrhizal fungi essential for seedling attachment. 0.001% reach maturity.
                         """),
                        ("Climate change impacts?",
                         """
                         Host vines flower earlier, mismatching parasite cycles. Rainfall changes
                         disrupt bud development. Higher temps reduce fly pollination efficiency.
                         Carbon starvation from host photosynthesis drops. No range shift possible.
                         """),
                        ("Biomimetic research?",
                         """
                         Camouflage tech from VOC mimicry. Gene-silencing mechanisms inform
                         pest control. Thermal regulation inspires smart textiles. Seed
                         adhesion properties studied for medical adhesives.
                         """),
                        ("Genetic mysteries?",
                         """
                         Missing circadian clock genes. Horizontal transfer from host confirmed.
                         Epigenetic controls unknown. RNA editing compensates for DNA simplicity.
                         Mitochondrial genome remains functional despite gene loss.
                         """),
                        ("How do conservationists monitor populations?",
                         """
                         Thermal drones detect blooms. DNA barcoding identifies species. Host
                         vine GIS mapping. Community reporting networks. Seed bank cryopreservation
                         attempts.
                         """),
                        ("Cultural revival efforts?",
                         """
                         Bloom festivals attract global tourists. Traditional uses documented
                         in ethnobotanical archives. Host vine planting initiatives. Artisan
                         crafts using dried flowers. Animated films spread awareness.
                         """),
                        ("Future extinction risks?",
                         """
                         Projected 100% loss by 2100 without intervention. Host vine extinction
                         cascades. Pollinator fly habitat destruction. Climate tipping points
                         passed. Lack of genetic diversity for adaptation.
                         """),
                        ("Last-ditch conservation measures?",
                         """
                         Grafting onto alternative hosts. Synthetic gene bank creation.
                         Artificial fly attractant pheromones. Host vine disease resistance
                         engineering. International trade bans enforced.
                         """),
                        ("How are octopuses' RNA edits tracking ocean changes?",
                         """
                         Octopus RNA now shows edited proteins for novel toxin resistance. Neural
                         voltage gates adapt to 0.3pH unit decreases. Kinesin edits repair
                         microplastic-damaged cells. Memory-forming edits accelerate learning
                         of new predator avoidance strategies in altered ecosystems.
                         """)
                    ]
                ),
                (
                    ("Why do humpback whales sing complex, evolving songs?",
                     """
                     Male humpbacks compose 30-minute songs using dual sound sources. Phrases 
                     repeat in hierarchical structures. Populations share regional dialects 
                     that evolve yearly. Songs communicate identity, breeding status, and 
                     navigation cues. Low frequencies (20-9,000Hz) travel 1,000km underwater.
                     """),
                    [
                        ("How do antlions build deadly sand traps?",
                         """
                         Larvae select dry, fine-grained sand. Spiral digging creates 60° slopes. 
                         Jaws flick sand to maximize avalanche effect. Vibrations mimic prey 
                         movements. Some species add pebble baffles to redirect escaping prey.
                         """),
                        ("Engineering precision details?",
                         """
                         Pit diameter optimized for local sand cohesion. Depth matches prey size
                         (3-10cm). Rotation speed (1rpm) maintains structural integrity.
                         Hydrophobic cuticle prevents moisture damage. UV-reflective particles
                         attract phototactic insects.
                         """),
                        ("How do larvae survive food scarcity?",
                         """
                         Metabolic rate drops to 5% normal. Fat reserves sustain 10-month fasts. 
                         Silk-lined pits reduce rebuild energy. Cannibalize smaller larvae. 
                         Vibrational mimicry steals others' prey.
                         """),
                        ("Evolutionary arms race with prey?",
                         """
                         Ants evolved backward escape jumps. Beetles detect pit vibrations.
                         Moths use powdered scales to reduce friction. Spiders build silk
                         bridges. Larvae counter with steeper angles and chemical lures.
                         """),
                        ("Ecological impacts?",
                         """
                         Control 40+ insect species populations. Pit microclimates host
                         specialized mites. Abandoned pits become plant nurseries. Larvae
                         enrich sand with nitrogenous waste. Birds use pits as dust baths.
                         """),
                        ("Biomimetic applications?",
                         """
                         Earthquake-resistant foundation designs. Solar concentrator mirrors
                         based on pit geometry. Avalanche prevention systems. Friction-reducing
                         surface textures. Autonomous construction robots.
                         """),
                        ("Lifecycle metamorphosis?",
                         """
                         Larval stage lasts 1-3 years. Pupal cocoons incorporate sand grains.
                         Adults emerge wingless, climb vegetation to unfurl wings. No feeding
                         stage - live 25 days to breed. Eggs laid in spiral patterns matching
                         pit construction.
                         """),
                        ("Climate vulnerability?",
                         """
                         Heavy rains collapse pits. Drought reduces prey activity. Temperature
                         extremes disrupt digging. Sand composition changes alter trap
                         efficiency. Invasive ant species avoid traps.
                         """),
                        ("Cultural significance?",
                         """
                         Aboriginal sand art mimics pit patterns. Ancient Greek "myrmex" coins.
                         Japanese gardens feature symbolic pits. European alchemy linked them
                         to earth elementals. Modern physics models fluid dynamics.
                         """),
                        ("Neuroscience insights?",
                         """
                         Vibration processing neural networks. Decision-making algorithms
                         during pit construction. Metabolic depression mechanisms. Sensory
                         integration of visual/mechanical cues. Instinct vs learning balance.
                         """),
                        ("How do larvae detect prey?",
                         """
                         Substrate vibrations sensed through leg receptors. Air pressure
                         changes detect wingbeats. Thermal sensors track body heat.
                         Chemoreceptors identify prey chemicals. Memory maps successful
                         pit locations.
                         """),
                        ("Conservation status?",
                         """
                         ️ 20% species endangered from habitat loss. Sand mining destroys
                         nesting areas. Artificial lighting disrupts predation. Invasive
                         species outcompete natives. Pesticides accumulate in food chain.
                         """),
                        ("Research techniques?",
                         """
                         3D-printed pit analogs test physics. High-speed video captures
                         prey capture. Laser vibrometry maps sand movement. Radioisotope
                         tagging tracks larval movements. DNA meta-barcoding identifies prey.
                         """),
                        ("Biomechanical limits?",
                         """
                         Maximum pit diameter=body length×π. Sand grain size <0.5mm for
                         effective avalanches. Jaw flick speed reaches 3m/s. Metabolic
                         ceiling allows 8 pit rebuilds/day. Silk production limited to
                         1mg/hour.
                         """),
                        ("Future robotics integration?",
                         """
                         Autonomous trap-building drones for pest control. Search/rescue
                         robots using vibration detection. Self-optimizing construction
                         algorithms. Energy-efficient digging mechanisms. Sand-stabilizing
                         nanomaterials.
                         """),
                        ("How are male humpbacks such a good singers?",
                         """
                    Male humpbacks compose very long 30-minute songs using two sound sources.Phrases
                    repeat in hierarchical structures.Populations are divided by regional dialects
                    that evolve continuously from year to year.Songs communicate identity, breeding status, and
                    navigation cues.They can also produce low-frequency sounds that can travel underwater
                        """)
                    ]
                )
            ]
            '''
        }

        if examples is not None:
            for steps, ex_str in examples:
                self.example_strs[steps] = ex_str

        if long_examples is not None:
            for steps, ex_str in long_examples:
                self.long_example_strs[steps] = ex_str

        self.examples_start = """
    ## FEW-SHOT EXAMPLES (Do not generate same or almost the same ones)
    ```python
    """

        self.examples_end = """
    ```
    """

        self.topics = [
            "Socotra Island's dragon blood trees",
            "Movile Cave's isolated ecosystem (Romania)",
            "Danakil Depression's acid pools (Ethiopia)",
            "Waitomo Glowworm Caves (New Zealand)",
            "Lençóis Maranhenses sand dunes (Brazil)",
            "Chocolate Hills formation (Philippines)",
            "Spotted Lake mineral deposits (Canada)",
            "Fly Geyser microbial colors (Nevada)",
            "Tianzi Mountain sandstone pillars (China)",
            "Giant's Causeway basalt columns (Ireland)",
            "Mantis shrimp's polarized vision",
            "Lyrebird vocal mimicry mechanisms",
            "Tardigrade cryptobiosis process",
            "Archerfish water-shooting accuracy",
            "Pistol shrimp shockwave hunting",
            "Bombardier beetle chemical defense",
            "Axolotl limb regeneration",
            "Narwhal tusk sensory capabilities",
            "Platypus electroreception",
            "Neutron star glitches",
            "Solar prominence dynamics",
            "Enceladus' cryovolcanic plumes",
            "Pāhoehoe lava formations",
            "Magnetar magnetic field strength",
            "Cosmic neutrino background",
            "Quantum foam fluctuations",
            "Spaghettification near black holes",
            "Solar sail navigation",
            "Dark matter halo structures",
            "Antikythera Mechanism functions",
            "Nazca Lines construction methods",
            "Moai transportation (Easter Island)",
            "Baghdad Battery controversy",
            "Roman concrete seawater hardening",
            "Viking sunstone navigation",
            "Saqqara Bird aerodynamics",
            "Göbekli Tepe carvings",
            "Voynich Manuscript analysis",
            "Olmec colossal heads",
            "Tensegrity bridge designs",
            "Aerogel insulation properties",
            "MEMS gyroscope manufacturing",
            "Hydrophobic surface engineering",
            "Metamaterial cloaking principles",
            "4D printing shape memory",
            "Piezoelectric shoe charging",
            "Self-healing concrete bacteria",
            "Quantum dot solar cells",
            "Starlink satellite avoidance",
            "Ainu bear ceremonies (Japan)",
            "Whistled language of La Gomera",
            "Mongolian throat singing",
            "Venetian glass bead making",
            "Sami reindeer herding",
            "Bhutanese archery traditions",
            "Navajo sandpainting",
            "Balinese water temples",
            "Maasai warrior jumps",
            "Korean hanji paper",
            "Cloud forests epiphyte networks",
            "Cenote freshwater layers (Mexico)",
            "Sargasso Sea floating ecosystem",
            "Hydrothermal vent communities",
            "Pantanal wetland seasons (Brazil)",
            "Peat bog preservation",
            "Atacama Desert lichens",
            "Waitoreke wetlands (New Zealand)",
            "Bioluminescent dinoflagellates",
            "Fire rainbow formation",
            "Halobacterium salt metabolism",
            "Gallium liquid properties",
            "Supercooled water stability",
            "Nitinol shape memory",
            "Chlorine trifluoride reactivity",
            "Ferrofluid manipulation",
            "Supercritical CO2 extraction",
            "Synesthesia neural pathways",
            "Blood-brain barrier",
            "Gut microbiome diversity",
            "Rapid eye movement sleep",
            "Hiccups evolutionary purpose",
            "Goosebumps mechanism",
            "Phantom limb sensations",
            "Taste bud regeneration",
            "Hypoxic training",
            "Blushing physiology",
            "Graphene conductivity",
            "Metallic glass formation",
            "Self-cleaning surfaces",
            "Carbon nanotube strength",
            "Ferrocrete durability",
            "Transparent aluminum",
            "Memory metal alloys",
            "Conductive polymers",
            "Glass frog transparency",
            "Mangrove salt filtration",
            "Sundarbans tiger adaptation",
            "Baobab water storage",
            "Carnivorous plant traps",
            "Electric eel discharge",
            "Leafcutter ant farming",
            "Spider web engineering",
            "Fire tornado dynamics",
            "Light pillars formation",
            "Morning glory clouds",
            "Volcanic lightning",
            "Snow roller formation",
            "Frost flowers growth",
            "Catatumbo lightning",
            "Sun dog phenomena",
            "Fire rainbow optics",
            "Virga precipitation",
            "Roman hypocaust heating",
            "Chinese seismograph",
            "Inca quipu records",
            "Persian qanat systems",
            "Egyptian faience",
            "Mayan astronomy",
            "Polynesian navigation",
            "Aboriginal firestick",
            "Memory palace technique",
            "Mnemonic major system",
            "Method acting",
            "Shadow puppetry",
            "Sand mandalas",
            "Kintsugi repair",
            "Wabi-sabi",
            "Ikebana principles",
            "Tea ceremony",
            "Origami mathematics",
            "Coral spawning events",
            "Jellyfish locomotion",
            "Cephalopod camouflage",
            "Spider silk production",
            "Ant colony algorithms",
            "Bee waggle dance",
            "Bat echolocation",
            "Elephant infrasound",
            "Whale fall ecosystems",
            "Quantum entanglement",
            "Superfluid helium",
            "Bose-Einstein condensate",
            "Quantum tunneling",
            "Casimir effect",
            "Sonoluminescence",
            "Triboluminescence",
            "Plasma crystals",
            "Optical tweezers",
            "CRISPR gene editing",
            "mRNA vaccine tech",
            "CAR-T cell therapy",
            "Synthetic biology",
            "Optogenetics",
            "Brain organoids",
            "Lab-grown meat",
            "Vertical farming",
            "Phytoremediation",
            "Mycoremediation",
            "Blockchain immutability",
            "Neural networks",
            "GAN image creation",
            "Quantum computing",
            "LiDAR mapping",
            "Hyperspectral imaging",
            "Synthetic aperture radar",
            "Photogrammetry",
            "Lidar archaeology",
            "Sonar mapping",
            "Perovskite solar",
            "Fusion reactor designs",
            "Tidal lagoon power",
            "Geothermal plants",
            "Piezoelectric roads",
            "Algae biofuels",
            "Hydrogen storage",
            "Carbon capture",
            "Wave energy",
            "Artificial photosynthesis",
            "Quokka smiling behavior (Australia)",
            "Pangolin scale composition",
            "Okapi stripe patterns",
            "Saiga antelope nasal structure",
            "Maned wolf communication",
            "Proboscis monkey nose function",
            "Aye-aye foraging techniques",
            "Sunda flying lemur gliding",
            "Thorny devil water collection",
            "Star-nosed mole sensory organs",
            "Mimic octopus impersonation",
            "Peacock spider mating dance",
            "Decorator crab camouflage",
            "Vampire squid bioluminescence",
            "Hagfish slime defense",
            "Nudibranch chemical warfare",
            "Sea butterfly swimming",
            "Velvet worm slime jets",
            "Water bear desiccation",
            "Trap-jaw ant mandibles",
            # Space & Astronomy
            "Neutron star mergers creating heavy elements",
            "Oumuamua interstellar object trajectory",
            "Kuiper Belt object composition",
            "Solar prominence magnetic dynamics",
            "Exoplanet atmospheric spectroscopy",
            "Dark energy's effect on universe expansion",
            "Pulsar navigation systems",
            "Space elevator material requirements",
            "Lunar lava tube habitats",
            "Mars regolith farming challenges",
            # Technology & Engineering
            "Metamaterial acoustic cloaking",
            "Self-healing asphalt microbial integration",
            "4D printed shape-memory polymers",
            "Neuromorphic computing chips",
            "Wireless power transmission efficiency",
            "Biohybrid robot muscle actuation",
            "Quantum radar principles",
            "Haptic feedback virtual reality gloves",
            "Transparent solar window tech",
            "Aerodynamic hyperloop pod designs",

            # Unique Biology
            "Axolotl limb regeneration genetics",
            "Tardigrade radiation resistance proteins",
            "Venus flytrap action potential mechanisms",
            "Platypus electroreception biology",
            "Naked mole-rat cancer immunity",
            "Lyrebird vocal mimicry neurology",
            "Chameleon color change nanostructures",
            "Electric eel voltage generation",
            "Ant mill death spiral behavior",
            "Slime mold problem-solving patterns",

            # Ancient History
            "Roman concrete seawater crystallization",
            "Inca quipu accounting system",
            "Viking sunstone navigation",
            "Egyptian faience production",
            "Chinese seismograph design",
            "Persian qanat irrigation",
            "Mayan astronomical codices",
            "Polynesian star navigation",
            "Native American controlled burns",
            "Indus Valley urban planning",

            # Art & Culture
            "Japanese kintsugi philosophy",
            "Aboriginal dot painting symbology",
            "Flamenco compás rhythmic patterns",
            "Balinese gamelan metallurgy",
            "Korean hanji papermaking",
            "Indian kolam rice art",
            "Celtic knotwork mathematics",
            "Sufi whirling meditation",
            "Hawaiian hula chant meaning",
            "Russian icon painting techniques",

            # Physics Phenomena
            "Superfluid helium quantum effects",
            "Magnetohydrodynamic wave propulsion",
            "Sonoluminescence bubble collapse",
            "Quantum locking in superconductors",
            "Ferrofluid spike formation",
            "Plasma filament formation",
            "Optical tweezers manipulation",
            "Casimir effect vacuum energy",
            "Ball lightning theories",
            "Non-Newtonian fluid behavior",

            # Geology & Earth Science
            "Antelope Canyon erosion patterns",
            "Danakil Depression extremophiles",
            "Bioluminescent cave ecosystems",
            "Penitente snow formations",
            "Lava lake convection currents",
            "Sailing stone movement",
            "Mineral fluorescence causes",
            "Petrified wood formation",
            "Desert varnish creation",
            "Thunder egg geode growth",

            # Medical Marvels
            "CRISPR gene editing accuracy",
            "mRNA vaccine development",
            "Prosthetic neural integration",
            "Organoid intelligence development",
            "Synthetic antibody production",
            "Telomere extension therapies",
            "Nanobot drug delivery",
            "Bionic eye resolution",
            "Phantom limb mapping",
            "Stem cell differentiation",

            # Plant Kingdom
            "Corpse flower pollination",
            "Strangler fig growth patterns",
            "Dragon's blood tree sap",
            "Bamboo flowering cycles",
            "Carnivorous pitcher plant",
            "Welwitschia leaf longevity",
            "Baobab water storage",
            "Resurrection plant revival",
            "Mimosa pudica movement",
            "Ghost plant chlorophyll",

            # Materials Science
            "Graphene conductivity",
            "Aerogel insulation",
            "Self-cleaning surfaces",
            "Memory metal alloys",
            "Transparent aluminum",
            "Hydrogel wound healing",
            "Carbon nanotube strength",
            "Bioactive glass regeneration",
            "Phase-change materials",
            "Photochromic lenses",

            # Animal Behavior
            "Bee waggle dance",
            "Octopus camouflage",
            "Wolf pack hierarchy",
            "Elephant infrasound",
            "Dolphin signature whistles",
            "Bird magnetic sensing",
            "Primate tool use",
            "Spider web engineering",
            "Firefly synchronization",
            "Penguin huddling",

            # Chemistry Wonders
            "Supercritical CO₂ extraction",
            "Triboluminescent sparks",
            "Liquid crystal phases",
            "Metallic hydrogen creation",
            "Photocatalytic water splitting",
            "Enzyme catalytic power",
            "Polymorph crystal forms",
            "Self-assembling molecules",
            "Stoichiometric ratios",
            "Catalytic converter",

            # Food Science
            "Maillard reaction",
            "Sourdough microbiome",
            "Chocolate tempering",
            "Umami taste receptors",
            "Molecular gastronomy",
            "Aged cheese crystallization",
            "Coffee roasting profiles",
            "Fermentation cultures",
            "Food dehydration",
            "Spice preservation",

            # Weather Phenomena
            "Morning glory clouds",
            "Derecho storm systems",
            "Diamond dust formation",
            "Fire whirl creation",
            "Virga precipitation",
            "Sundog refraction",
            "Haboob dust walls",
            "Mammatus cloud",
            "Heat lightning",
            "Thundersnow",

            # Ancient Engineering
            "Egyptian pyramid alignment",
            "Roman aqueduct gradients",
            "Chinese earthquake detector",
            "Incan suspension bridges",
            "Persian windcatchers",
            "Mayan water filtration",
            "Greek fire composition",
            "Indian stepwells",
            "Native American earthworks",
            "Viking shipbuilding",

            # Forensic Science
            "Entomological timelines",
            "DNA phenotyping",
            "Gunshot residue",
            "Blood spatter",
            "Fingerprint lifting",
            "Toxicology screening",
            "Document authentication",
            "Facial reconstruction",
            "Soil analysis",
            "Tool mark",

            # Energy Tech
            "Fusion plasma",
            "Tidal generators",
            "Geothermal plants",
            "Biofuel algae",
            "Piezoelectric roads",
            "Hydrogen storage",
            "Thorium reactors",
            "Solar paint",
            "Wave energy",
            "Gravity storage",

            # Neuroscience
            "Neuroplasticity",
            "Mirror neurons",
            "Brain mapping",
            "Memory consolidation",
            "Synaptic pruning",
            "Pain perception",
            "Consciousness theories",
            "Sleep cycles",
            "Déjà vu",
            "Phantom smells",

            # Paleontology
            "Feathered dinosaurs",
            "Amber preservation",
            "Trace fossils",
            "Mass extinction",
            "DNA recovery",
            "Coprolite analysis",
            "Evolutionary missing links",
            "Cambrian explosion",
            "Fossil dating",
            "Paleoclimate",

            # Architecture
            "Biomimetic design",
            "Earthquake retrofitting",
            "Green roofs",
            "Tensegrity structures",
            "Passive heating",
            "Acoustic engineering",
            "3D-printed buildings",
            "Feng shui principles",
            "Gothic vaulting",
            "Bamboo construction"
        ]

    def get_follow_ups_format(self, steps: int, mode: str = 'multi'):
        if mode == 'long':
            format_follow_ups = f"""
              [
                  ("[Follow-up 1: **topic two**]", "[Answer 1: **topic two**]"),
                  ("[Follow-up 2: **topic two**]", "[Answer 2: **topic two**]"),
                  # ... {steps - 1} total: **topic two**
                  ("[Final follow-up {steps}: **topic one**]", "[Final answer {steps}: **topic one**]"),
              ]
        """
        elif steps == 1:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
              ]
        """
        elif steps == 2:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
                  ("[Follow-up 2]", "[Answer 2]"),
              ]
        """
        else:
            format_follow_ups = f"""
              [
                  ("[Follow-up 1]", "[Answer 1]"),
                  ("[Follow-up 2]", "[Answer 2]"),
                  # ... {steps} total
              ]
        """
        return format_follow_ups

    def get_description(self, steps: int, num_examples: int, prior_steps: int, mode: str = 'multi'):
        return f"""
    # Memory Reinforcement Learning Data Generation - {steps} Step(s)

    ## TASK DESCRIPTION
    Generate {num_examples} examples of interconnected interactions testing multi-step memory retention. They will be
    used to train reactive models, that works in real-time processing mode - processing only the single messages (interactions),
    and keeps conversation history in short-term memory.

    Each example must:

    1. Start with initial factual QA (geography/science/history) - query and answer
    2. Contain {steps} follow-up QAs progressively referencing prior answers - list of queries and answers
    3. Each answer introduces 3-4 new facts while repeating 3+ previous facts
    4. Final interaction requires synthesizing info from {prior_steps}+ prior steps
    ## FORMAT REQUIREMENTS
    ```python
    [
        (
            ("[Initial Question]", "[Initial Answer]"),
            {self.get_follow_ups_format(steps, mode)}
        ),
        # {num_examples} total entries
    ]
    ```
    """ if mode == 'multi' else f"""
    # Memory Reinforcement Learning Data Generation - {steps} Step(s) - Long-Range Strategy

    ## TASK DESCRIPTION
    Generate {num_examples} examples of interconnected interactions testing long-range and multi-step memory retention. They
    will be used to train reactive models, that works in real-time processing mode - processing only the single messages (interactions),
    and keeps conversation history in short-term memory.
    
    In Long-Range strategy each generated example should contain two completely different message **topics**:
    - first topic is used only in first and last interaction
    - second topic is used in all interactions between first and last

    Each example must:

    1. Start with initial factual QA (geography/science/history/others) - query and answer - for the first **topic**
    2. Contain {steps} follow-up QAs list of queries and answers
    3. First query in follow-up QAs (interactions) is changing **topic** into second one - not connected to first QA, and answer is also for new **topic**
    4. All next follow-up interactions, except the last one, have to continue the second **topic**:
        - Each answer introduces 3-4 new facts while repeating 3+ previous facts
    4. Final, last interaction is going back to first **topic** - asking a question about data from first QA pair (interaction) and the answer is containing most facts from first interaction
    ## FORMAT REQUIREMENTS
    ```python
    [
        (
            ("[Initial Question: **topic one**]", "[Initial Answer: **topic one**]"),
            {self.get_follow_ups_format(steps, mode)}
        ),
        # {num_examples} total entries
    ]
    ```
    """

    def get_critical_rules(self, prior_steps: int, mode: str = 'multi'):
        if mode == 'multi':
            return f"""
        ## CRITICAL RULES
        1. Factual Consistency:
          - All geographical/scientific facts must be accurate
          - No fictional elements - only real-world entities
          - Real-world knowledge, but only a basic facts like in examples
          - Simple facts in form of stories
          - Shouldn't include a lot of numbers - only basic facts
          - No calculations, etc., only the knowledge and just dialog content
        2. Non-formal language:
          - Model is pre-trained on stories with non-formal language - it should be compatible
          - While dataset includes real world facts, they should be served in story-like format
        3. Referential Integrity:
          - Each follow-up references 3+ facts from previous answers
          - Final answer must combine facts from {prior_steps} previous steps
        4. Progressive Retention:
          - Each answer is on the same topic
          - Knowledge is accumulated in answers
          - Queries reference previous answers
        5. Do not repeat examples
          - each item should be unique - do not use the same examples as provided
          - please do not generate same examples as in FEW SHOTS items should be new
          - don't generate examples in same topics as in FEW SHOT EXAMPLES, like Mount Everest, Nile description, Amazon forest, Grand Canyon, Coral Reef, Sequoia Trees or Great Wall
          - be creative for topics
          - try a lot different topics
        """
        else:
            return f"""
        ## CRITICAL RULES
        1. Factual Consistency:
          - All geographical/scientific facts must be accurate
          - No fictional elements - only real-world entities
          - Real-world knowledge, but only a basic facts like in examples
          - Simple facts in form of stories
          - Shouldn't include a lot of numbers - only basic facts
          - No calculations, etc., only the knowledge and just dialog content
        2. Non-formal language:
          - Model is pre-trained on stories with non-formal language - it should be compatible
          - While dataset includes real world facts, they should be served in story-like format
        3. Referential Integrity:
          - Each middle follow-up for **topic two** references 3+ facts from previous answers and trying to combine facts from {prior_steps} previous steps (except first step)
          - Final answer must change back to **topic one** and reference 3+ facts from initial QA pair/interaction
        4. Progressive Retention:
          - Each answer from the middle of sequence (except initial and last follow-up) is on the same topic (**topic two**)
          - Knowledge is accumulated in answers
          - Queries reference previous answers of the same **topic**
        5. Do not repeat examples
          - each item should be unique - do not use the same examples as provided
          - please do not generate same examples as in FEW SHOTS items should be new
          - don't generate examples in same topics as in FEW SHOT EXAMPLES, like Mount Everest, Nile description, Amazon forest, Grand Canyon, Coral Reef, Sequoia Trees or Great Wall
          - be creative for topics
          - try a lot different topics
      """

    def get_examples(self, steps: int, mode: str = 'multi'):
        exs = self.example_strs[steps] if mode == 'multi' else self.long_example_strs[steps]
        return self.examples_start + exs + self.examples_end

    def get_topics(self, num_topics: int, mode: str = 'multi'):
        topics = random.choices(self.topics, k=num_topics)

        if num_topics != 0:
            topics_bullets = '/n'.join([f'- {topic}' for topic in topics])

            topics_str = f"""
      ## TOPICS FOR GENERATED EXAMPLES:
        - do not use same examples like FEW SHOTS, try different topics, like following ones
        - you can use one of the following topics for generated examples or similar one
        {topics_bullets}
      """ if mode == 'multi' else f"""
      ## TOPICS FOR GENERATED EXAMPLES:
      - do not use same examples like FEW SHOTS, try different **topics**, like following ones
      - you should use **TWO** of the following **topics** for generated examples or some similar **topics**
      {topics_bullets}
      """
        else:
            topics_str = ''

        return topics_str

    def get_final_instructions(self, steps: int, num_examples: int, include_no_think: bool = True, mode: str = 'multi'):
        instructions = f"""
        ## GENERATION INSTRUCTIONS
        1. Start each sequence with geographical/scientific fact
        2. Design {steps} follow-ups requiring cumulative understanding
        3. Ensure final answer combines numerical, temporal, and spatial data
        4. Validate all  factual accuracy
        ## OUTPUT VALIDATION
        - No placeholder text ("...")
        - All facts consistent
        - {steps} follow-ups per entry
        - Final answer combines 3+ prior facts
        - Output only the final list - without wrapping it with '```python' and '```'
        - Output is a single list of tuples in same format as examples
        - Output contains {num_examples} elements with the same format as examples
        - Do not generate separate list for each example, only single list of tuples
        
        Generate {num_examples} entries following EXACTLY this structure.
        """ if mode == 'multi' else f"""
        ## GENERATION INSTRUCTIONS
        1. Start each sequence with geographical/scientific fact with **topic one**
        2. Design {steps - 1} follow-ups requiring cumulative understanding for **topic two**
        3. Ensure final query is switching back to **topic one** and final answer is including data from initial QA pair
        4. Validate all factual accuracy
        ## OUTPUT VALIDATION
        - No placeholder text ("...")
        - All facts consistent
        - {steps} follow-ups per entry
        - Final answer combines 3-4+ facts from first initial interaction
        - Output only the final list - without wrapping it with '```python' and '```'
        - Output is a single list of tuples in same format as examples
        - Output contains {num_examples} elements with the same format as examples
        - Do not generate separate list for each example, only single list of tuples
        
        Generate {num_examples} entries following EXACTLY this structure.
        """

        no_think = """
        /no_think
        """

        return instructions + no_think if include_no_think else instructions

    def get_prior_steps(self, steps: int):
        if steps < 4:
            prior_steps = 1
        elif steps >= 4:
            prior_steps = 2
        elif steps <= 8:
            prior_steps = 3
        else:
            prior_steps = 4

        return prior_steps

    def get_system_prompt(self, num_examples: int):
        return f"""
        You are a Memory Reinforcement Learning dataset generator for a Reactive Transformer model.
        Output must be a Python list of tuples containing initial query-answer pairs and list of follow-up interactions, without any code, explanation, or metadata.
        You are used to generate list of exactly {num_examples} tuples with three elements: question, answer and list of follow-up interactions, based on real world knowledge in form of simple stories
        """

    def __call__(self, steps: int, num_examples: int = 10, num_topics: int = 10, mode: str = 'multi',
                 include_no_think: bool = True):
        prior_steps = self.get_prior_steps(steps)
        return (self.get_description(steps, num_examples, prior_steps, mode=mode) +
                self.get_critical_rules(prior_steps, mode=mode) +
                self.get_examples(steps, mode=mode) + self.get_topics(num_topics, mode=mode) +
                self.get_final_instructions(steps, num_examples, include_no_think=include_no_think, mode=mode))


class MrlSyntheticDatasetGenerator:
    def __init__(
            self, max_items: int = None, model_name: str = "qwen/qwen3-4b-fp8",
            api_url: str = "https://api.novita.ai/v3/openai",
            api_key: str = "sk_QnQVgESsna-bHJPSRR-UbYVfSYtoa77E7T4KdFtnTag"
    ):
        self.failed_count = 0
        self.items = self._init_items()
        self.max_items = max_items
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
        )
        self.model_name = model_name

    def _init_items(self):
        return { 'query': [], 'answer': [], 'interactions': [] }

    def generate_items(
            self, prompt: str, stream: bool = False, temperature: float = 0.7,
            top_p: float = 0.9, top_k: int = 50, max_tokens: int = 15000,
            system_prompt: str = "", timeout: int = 120
    ):
        try:
            presence_penalty = 0
            frequency_penalty = 0
            repetition_penalty = 1
            min_p = 0
            response_format = {"type": "text"}

            chat_completion_res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                extra_body={
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "min_p": min_p
                },
                timeout=timeout
            )

            if stream:
                t1 = datetime.timestamp(datetime.now())
                acc = ''
                for chunk in chat_completion_res:
                    if datetime.timestamp(datetime.now()) - t1 > timeout:
                        break
                    ch = chunk.choices[0].delta.content or ""
                    print(ch, end="")
                    acc += ch
                return acc

            return chat_completion_res.choices[0].message.content
        except Exception as e:
            print('API Error', e)
            return '[]'

    def process_interactions(self, interactions: list[tuple[str, str]]):
        return [{'query': query.strip(), 'answer': answer.strip()} for query, answer in interactions]

    def process_items(self, items_str: str, steps: int, stream: bool = False) -> int:
        try:
            if not items_str.rstrip().endswith(']'):
                items_str = items_str.rstrip() + ']'
            if not items_str.lstrip().startswith('[]'):
                items_str = '[' + items_str.lstrip()
            if not stream:
                print(items_str)
            items_list = eval(items_str)
            items_len = len(items_list)
            for (query, answer), interactions in items_list:
                if query is not None and answer is not None and interactions is not None and len(interactions) == steps:
                    self.items['query'].append(query.strip())
                    self.items['answer'].append(answer.strip())
                    self.items['interactions'].append(self.process_interactions(interactions))
            return items_len
        except:
            self.failed_count += 1
            print(f'Cannot process generated list! Failed {self.failed_count} times')
            return 0

    def get_dataset(self) -> Dataset:
        return Dataset.from_dict(self.items)

    def __call__(self, prompt_creator: MrlPromptCreator, steps: int, iterations: int, num_examples: int = 10,
                 num_topics: int = 10, include_no_think: bool = True, mode: str = 'multi', stream: bool = False,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, max_tokens: int = 15000,
                 timeout: int = 120, restart: bool = False):
        if restart:
            self.items = self._init_items()

        for iteration in range(iterations):
            # Create example prompt
            prompt = prompt_creator(steps, num_examples=num_examples, num_topics=num_topics,
                                    include_no_think=include_no_think, mode=mode)
            # Get system prompt
            system_prompt = prompt_creator.get_system_prompt(num_examples)
            # Call API to generate items
            txt = self.generate_items(
                prompt, stream=stream, temperature=temperature, top_p=top_p,
                top_k=top_k, max_tokens=max_tokens, timeout=timeout, system_prompt=system_prompt,
            )
            new_items_len = self.process_items(txt[21:] if '<think>' in txt else txt, steps, stream=stream)
            total_items = len(self.items['query'])
            if stream:
                print('\n')
            print(f'{iteration + 1}/{iterations}: Added {new_items_len} new items, total items {total_items}')
            if total_items > self.max_items:
                print('Max items limit reached, breaking.')
                break



class MrlGeneratorPostprocessor:
    def __init__(self, generator: MrlSyntheticDatasetGenerator, dataset_id: str, config_name: str, split: str = 'train',
                 token: str = None):
        self.generator = generator
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.split = split
        self.token = token

    def filter_duplicates(self):
        queries = []
        answers = []
        interactions = []

        items_len = len(self.generator.items['query'])

        print(f'Original size: {items_len}')

        for i in range(items_len):
            query = self.generator.items['query'][i]
            if not query in queries:
                queries.append(query)
                answers.append(self.generator.items['answer'][i])
                interactions.append(self.generator.items['interactions'][i])

        print(f'Filtered size: {len(queries)}')

        self.generator.items = {
            'query': queries,
            'answer': answers,
            'interactions': interactions
        }

    def remove_incorrect_interactions(self, steps: int):
        queries = []
        answers = []
        interactions = []

        counts = {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
        }

        items_len = len(self.generator.items['query'])

        print(f'Original size: {items_len}')

        for i in range(items_len):
            item_inters = self.generator.items['interactions'][i]
            item_inters_len = len(item_inters)
            counts[item_inters_len] += 1

            if item_inters_len == steps:
                queries.append(self.generator.items['query'][i])
                answers.append(self.generator.items['answer'][i])
                interactions.append(item_inters)

        print(f"Unique interaction len counts: {[(l, c) for l, c in counts.items() if c > 0]}")

        print(f'Filtered size: {len(queries)}')

        self.generator.items = {
            'query': queries,
            'answer': answers,
            'interactions': interactions
        }

    def get_subset(self, split_idx: int) -> MrlSyntheticDatasetGenerator:
        queries_a, queries_b = self.generator.items['query'][:split_idx], self.generator.items['query'][split_idx:]
        answers_a, answers_b = self.generator.items['answer'][:split_idx], self.generator.items['answer'][split_idx:]
        interactions_a, interactions_b = self.generator.items['interactions'][:split_idx], self.generator.items['interactions'][split_idx:]

        self.generator.items = {
            'query': queries_a,
            'answer': answers_a,
            'interactions': interactions_a,
        }

        generator_b = MrlSyntheticDatasetGenerator(max_items=self.generator.max_items)

        generator_b.items = {
            'query': queries_b,
            'answer': answers_b,
            'interactions': interactions_b,
        }

        return generator_b

    def append_from_existing_dataset(self):
        dataset = load_dataset(self.dataset_id, self.config_name, split=self.split, token=self.token)

        self.generator.items = {
            'query': dataset['query'] + self.generator.items['query'],
            'answer': dataset['answer'] + self.generator.items['answer'],
            'interactions': dataset['interactions'] + self.generator.items['interactions']
        }

    def push_to_hf_hub(self):
        ds = self.generator.get_dataset()
        ds.push_to_hub(repo_id=self.dataset_id, config_name=self.config_name, split=self.split, token=self.token)
