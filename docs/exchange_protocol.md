# Agent Exchange Protocol
An Agent's Compartment device supports an item exchange protocol, enabling direct trade between Agents.

To begin, an Agent defines exchange rules that specify which items and quantities can be offered or requested. The Agent then broadcasts a protocol signal to discover active exchange rules from nearby Agents. Once a desirable offer is found, the Agent can approach the target and execute the exchange according to the established rules.


## Actions
- **update_exchange**: This method is used to define or update Agent's exchange rules. You can declare which items and quantities you are willing to offer, as well as the types and quantities of items you are willing to accept in return.
- **list_exchange**: Initiates an exchange protocol broadcast and retrieves the currently active exchange rules from all nearby Agents within a 20-meter range who support this protocol. Each line in the response represents one active exchange rule, for example: 'Agent 12|Gear,1;Lumen Fungus,1->Battery Pack,1', 'Agent 33|Mushroom,1->NONE'
- **execute_exchange**: Attempts an atomic item swap with the target Agent according to their active exchange rules. Accepts a single exchange pair per call.



## Exchange Protocol
```
ExchangeList   = { RuleLine , LineBreak } ;

RuleLine       = AgentID , OptWS , '|' , OptWS , Offer , OptWS , '->' , OptWS , Request , OptWS ;

AgentID        = Identifier ;

Offer          = 'NONE' | ItemList ;
Request        = 'NONE' | ItemList ;

ItemList       = Item , { OptWS , ';' , OptWS , Item } ;

Item           = ItemName , OptWS , ',' , OptWS , Quantity ;

ItemName       = ItemChar , { ItemChar } ;
ItemChar       = Letter | Digit | ' ' | '_' | '-' | '.' | '/' ;

Quantity       = NonZeroDigit , { Digit } ;

Identifier     = IdentChar , { IdentChar } ;
IdentChar      = Letter | Digit | '_' | '-' ;

Letter         = 'A'..'Z' | 'a'..'z' ;
Digit          = '0'..'9' ;
NonZeroDigit   = '1'..'9' ;

OptWS          = { ' ' } ;
LineBreak      = '\n' | '\r\n' ;
```

## Example
```
Agent-07 | Aether Steel,2 -> Flux Titanium,1
Agent-12 | Battery Pack,1 -> ANY,2
Agent-15 | Power Cell Large,3; Flux Coil,1 -> NONE
```