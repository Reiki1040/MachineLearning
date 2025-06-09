name=startetrobo_mac.command; if [ ! -f $name ]; then curl -O https://raw.githubusercontent.com/ETrobocon/etrobo/master/scripts/$name; chmod +x ~/$name; fi; ~/$name
